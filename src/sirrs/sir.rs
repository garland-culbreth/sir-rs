//! Three compartment SIR model and methods.
//!
//! Allows transition rates:
//!  - S → I  
//!  - I → R  
//!  - R → S  
use faer::{Mat, traits::num_traits::ToPrimitive};

/// Numerical integrator variables
///
/// This private struct exists to make indexing k and y during integration
/// simpler.
struct SystemVars {
    s: f64,
    i: f64,
    r: f64,
}

/// Create and run an SIR model.
pub struct Model {
    /// Number of indices to generate and solve. The length of the series.
    pub length: usize,
    /// Size of integration step.
    pub step_size: f64,
    /// Initial infectious population fraction.
    pub i_popf_init: f64,
    /// Initial removed population fraction.
    pub r_popf_init: f64,
    /// Transition rate from S into I. Must be in [0, 1].
    pub incidence_rate: f64,
    /// Transition rate from I into R. Must be in [0, 1].
    pub removal_rate: f64,
    /// Transition rate from I into S. Must be in [0, 1].
    pub recovery_rate: f64,
    /// Susceptible population fraction at each index. 1D Array with `length` number of elements.
    pub s_popf: Mat<f64>,
    /// Inectious population fraction at each index. 1D Array with `length` number of elements.
    pub i_popf: Mat<f64>,
    /// Removed population fraction at each index. 1D Array with `length` number of elements.
    pub r_popf: Mat<f64>,
}

impl Model {
    /// Initialize population fractions. Creates arrays of length `self.length`
    /// to store the population fractions at each index and sets the 0th index
    /// of each equal to the corresponding initial population fraction.
    pub fn init_popf(&mut self) -> &mut Model {
        let n_steps = (self.length.to_f64().unwrap() / self.step_size)
            .to_usize()
            .unwrap();
        self.s_popf = Mat::zeros(n_steps, 1);
        self.i_popf = Mat::zeros(n_steps, 1);
        self.r_popf = Mat::zeros(n_steps, 1);
        let s_init = 1.0 - self.i_popf_init - self.r_popf_init; // Population fractions must sum to 1.
        self.s_popf[(0, 0)] = s_init;
        self.i_popf[(0, 0)] = self.i_popf_init;
        self.r_popf[(0, 0)] = self.r_popf_init;
        return self;
    }

    fn dsdt(&self, susceptible: f64, infectious: f64) -> f64 {
        return (-self.incidence_rate * susceptible * infectious)
            + (self.recovery_rate * infectious);
    }

    fn didt(&self, susceptible: f64, infectious: f64) -> f64 {
        return (self.incidence_rate * susceptible * infectious)
            - ((self.recovery_rate + self.removal_rate) * infectious);
    }

    fn drdt(&self, infectious: f64) -> f64 {
        return self.removal_rate * infectious;
    }

    /// Run the SIR differential equations by the first-order euler method.
    ///
    /// This solution method is very rough and only suitable for demonstration.
    pub fn run_euler(&mut self) -> &Model {
        let h = self.step_size;
        let n = (self.length.to_f64().unwrap() / h)
            .ceil()
            .to_usize()
            .unwrap();
        for i in 0..n - 1 {
            let ds = self.dsdt(self.s_popf[(i, 0)], self.i_popf[(i, 0)]);
            let di = self.didt(self.s_popf[(i, 0)], self.i_popf[(i, 0)]);
            let dr = self.drdt(self.i_popf[(i, 0)]);
            self.s_popf[(i + 1, 0)] = self.s_popf[(i, 0)] + (h * ds);
            self.i_popf[(i + 1, 0)] = self.i_popf[(i, 0)] + (h * di);
            self.r_popf[(i + 1, 0)] = self.r_popf[(i, 0)] + (h * dr);
            println!(
                "t={}: s={:.6} i={:.6} r={:.6}",
                i,
                self.s_popf[(i, 0)],
                self.i_popf[(i, 0)],
                self.r_popf[(i, 0)]
            );
        }
        return self;
    }

    /// Construct array of runge-kutta intermediate values for each variable.
    fn init_y(&self) -> [SystemVars; 5] {
        return [
            SystemVars {
                s: 0.0,
                i: 0.0,
                r: 0.0,
            },
            SystemVars {
                s: 0.0,
                i: 0.0,
                r: 0.0,
            },
            SystemVars {
                s: 0.0,
                i: 0.0,
                r: 0.0,
            },
            SystemVars {
                s: 0.0,
                i: 0.0,
                r: 0.0,
            },
            SystemVars {
                s: 0.0,
                i: 0.0,
                r: 0.0,
            },
        ];
    }

    /// Construct array of runge-kutta constants for each variable.
    fn init_k(&self) -> [SystemVars; 5] {
        return [
            SystemVars {
                s: 0.0,
                i: 0.0,
                r: 0.0,
            },
            SystemVars {
                s: 0.0,
                i: 0.0,
                r: 0.0,
            },
            SystemVars {
                s: 0.0,
                i: 0.0,
                r: 0.0,
            },
            SystemVars {
                s: 0.0,
                i: 0.0,
                r: 0.0,
            },
            SystemVars {
                s: 0.0,
                i: 0.0,
                r: 0.0,
            },
        ];
    }

    /// Construct array of step sizes corresponding to each runge-kutta order.
    fn init_h(&self) -> [f64; 4] {
        return [
            self.step_size / 2.0,
            self.step_size / 2.0,
            self.step_size,
            self.step_size,
        ];
    }

    /// Compute a runge-kutta approximate function value.
    fn next_y(&self, y: f64, k: f64, h: f64) -> f64 {
        return y + (k * h);
    }

    /// Compute a 4th order runge-kutta time step for the system.
    fn rk4_step(&self, t: usize) -> [SystemVars; 5] {
        let mut y = self.init_y();
        let mut k = self.init_k();
        let h = self.init_h();
        y[0].s = self.s_popf[(t, 0)];
        y[0].i = self.i_popf[(t, 0)];
        y[0].r = self.r_popf[(t, 0)];
        for i in 0..4 {
            k[i + 1].s = self.dsdt(y[i].s, y[i].i);
            k[i + 1].i = self.didt(y[i].s, y[i].i);
            k[i + 1].r = self.drdt(y[i].i);
            y[i + 1].s = self.next_y(y[0].s, k[i + 1].s, h[i]);
            y[i + 1].i = self.next_y(y[0].i, k[i + 1].i, h[i]);
            y[i + 1].r = self.next_y(y[0].r, k[i + 1].r, h[i]);
        }
        return k;
    }

    /// Solve the system by the 4th order Runge-Kutta method.
    ///
    /// This method is suitable for general purposes.
    pub fn run_rk4(&mut self) -> &Model {
        let n = (self.length.to_f64().unwrap() / self.step_size)
            .ceil()
            .to_usize()
            .unwrap();
        for t in 0..n - 1 {
            let k = self.rk4_step(t);
            let ds = (k[1].s + (2.0 * k[2].s) + (2.0 * k[3].s) + k[4].s) * (self.step_size / 6.0);
            let di = (k[1].i + (2.0 * k[2].i) + (2.0 * k[3].i) + k[4].i) * (self.step_size / 6.0);
            let dr = (k[1].r + (2.0 * k[2].r) + (2.0 * k[3].r) + k[4].r) * (self.step_size / 6.0);
            self.s_popf[(t + 1, 0)] = self.s_popf[(t, 0)] + ds;
            self.i_popf[(t + 1, 0)] = self.i_popf[(t, 0)] + di;
            self.r_popf[(t + 1, 0)] = self.r_popf[(t, 0)] + dr;
            if t % 10 == 0 {
                println!(
                    "t={:.1} s={:.6} i={:.6} r={:.6}",
                    t.to_f64().unwrap() * self.step_size,
                    self.s_popf[(t, 0)],
                    self.i_popf[(t, 0)],
                    self.r_popf[(t, 0)],
                );
            }
        }
        return self;
    }
}

#[cfg(test)]
mod tests {
    use crate::sirrs::sir::Model;
    use faer::{Mat, traits::num_traits::ToPrimitive};

    #[test]
    fn test_init_model() {
        let model: Model = Model {
            length: 10,
            step_size: 1.0,
            i_popf_init: 0.01,
            r_popf_init: 0.0,
            incidence_rate: 0.02,
            removal_rate: 0.03,
            recovery_rate: 0.04,
            s_popf: Mat::new(),
            i_popf: Mat::new(),
            r_popf: Mat::new(),
        };
        assert_eq!(
            model.length, 10,
            "Bad length, expected 10 got {}",
            model.length
        );
        assert_eq!(
            model.i_popf_init, 0.01,
            "Bad i_popf_init, expected 0.01 got {}",
            model.i_popf_init,
        );
        assert_eq!(
            model.r_popf_init, 0.0,
            "Bad r_popf_init, expected 0.0 got {}",
            model.r_popf_init,
        );
        assert_eq!(
            model.incidence_rate, 0.02,
            "Bad incidence_rate, expected 0.02 got {}",
            model.incidence_rate,
        );
        assert_eq!(
            model.removal_rate, 0.03,
            "Bad , expected 0.03 got {}",
            model.removal_rate,
        );
        assert_eq!(
            model.recovery_rate, 0.04,
            "Bad , expected 0.04 got {}",
            model.recovery_rate,
        );
        assert_eq!(
            model.s_popf,
            Mat::new(),
            "Bad , expected Mat::new() got {:?}",
            model.s_popf,
        );
        assert_eq!(
            model.i_popf,
            Mat::new(),
            "Bad , expected Mat::new() got {:?}",
            model.i_popf,
        );
        assert_eq!(
            model.r_popf,
            Mat::new(),
            "Bad , expected Mat::new() got {:?}",
            model.r_popf,
        );
    }

    #[test]
    fn test_init_popf() {
        let mut model: Model = Model {
            length: 10,
            step_size: 1.0,
            i_popf_init: 0.01,
            r_popf_init: 0.0,
            incidence_rate: 0.02,
            removal_rate: 0.03,
            recovery_rate: 0.04,
            s_popf: Mat::new(),
            i_popf: Mat::new(),
            r_popf: Mat::new(),
        };
        model.init_popf();
        assert_eq!(
            model.s_popf.shape(),
            (model.length, 1),
            "Bad s_popf dimensions, expected {:?} got {:?}.",
            (model.length, 1),
            model.s_popf.shape(),
        );
        assert_eq!(
            model.i_popf.shape(),
            (model.length, 1),
            "Bad i_popf dimensions, expected {:?} got {:?}.",
            (model.length, 1),
            model.i_popf.shape(),
        );
        assert_eq!(
            model.r_popf.shape(),
            (model.length, 1),
            "Bad r_popf dimensions, expected {:?} got {:?}.",
            (model.length, 1),
            model.r_popf.shape(),
        );
        assert_eq!(
            model.s_popf[(0, 0)],
            1.0 - model.i_popf_init - model.r_popf_init,
            "Bad s_popf[0] initialization value, expected {} got {}.",
            1.0 - model.i_popf_init - model.r_popf_init,
            model.s_popf[(0, 0)]
        );
        assert_eq!(
            model.i_popf[(0, 0)],
            model.i_popf_init,
            "Bad i_popf[0] initialization value, expected {} got {}.",
            model.i_popf_init,
            model.i_popf[(0, 0)],
        );
        assert_eq!(
            model.r_popf[(0, 0)],
            model.r_popf_init,
            "Bad r_popf[0] initialization value, expected {} got {}.",
            model.r_popf_init,
            model.r_popf[(0, 0)],
        );
        for t in 1..model.length {
            assert_eq!(
                model.s_popf[(t, 0)],
                0.0,
                "Bad s_popf[t>0] initialization value, expected 0.0 got {}.",
                model.s_popf[(t, 0)]
            );
            assert_eq!(
                model.i_popf[(t, 0)],
                0.0,
                "Bad i_popf[t>0] initialization value, expected 0.0 got {}.",
                model.i_popf[(t, 0)]
            );
            assert_eq!(
                model.r_popf[(t, 0)],
                0.0,
                "Bad r_popf[t>0] initialization value, expected 0.0 got {}.",
                model.r_popf[(t, 0)]
            );
        }
    }

    #[test]
    fn test_run_euler() {
        let mut model: Model = Model {
            length: 10,
            step_size: 1.0,
            i_popf_init: 0.01,
            r_popf_init: 0.0,
            incidence_rate: 0.02,
            removal_rate: 0.03,
            recovery_rate: 0.04,
            s_popf: Mat::new(),
            i_popf: Mat::new(),
            r_popf: Mat::new(),
        };
        model.init_popf();
        model.run_euler();
        let h = model.step_size;
        let n = (model.length.to_f64().unwrap() / h)
            .ceil()
            .to_usize()
            .unwrap();
        for t in 1..n - 1 {
            let dsdt = model.dsdt(model.s_popf[(t - 1, 0)], model.i_popf[(t - 1, 0)]);
            let didt = model.didt(model.s_popf[(t - 1, 0)], model.i_popf[(t - 1, 0)]);
            let drdt = model.drdt(model.i_popf[(t - 1, 0)]);
            model.s_popf[(t, 0)] = model.s_popf[(t - 1, 0)] + (h * dsdt);
            model.i_popf[(t, 0)] = model.i_popf[(t - 1, 0)] + (h * didt);
            model.r_popf[(t, 0)] = model.r_popf[(t - 1, 0)] + (h * drdt);
            assert!(
                (model.s_popf[(t, 0)] >= 0.0) & (model.s_popf[(t, 0)] <= 1.0),
                "s_popf[(t, 0)] not in [0, 1] at time {}, got {}",
                t,
                model.s_popf[(t, 0)]
            );
            assert!(
                (model.i_popf[(t, 0)] >= 0.0) & (model.i_popf[(t, 0)] <= 1.0),
                "i_popf[(t, 0)] not in [0, 1] at time {}, got {}",
                t,
                model.i_popf[(t, 0)]
            );
            assert!(
                (model.r_popf[(t, 0)] >= 0.0) & (model.r_popf[(t, 0)] <= 1.0),
                "r_popf[(t, 0)] not in [0, 1] at time {}, got {}",
                t,
                model.r_popf[(t, 0)]
            );
            assert_eq!(
                model.s_popf[(t, 0)],
                model.s_popf[(t - 1, 0)] + (h * dsdt),
                "Bad s_popf[(t, 0)] at time {}, expected {} got {}",
                t,
                model.s_popf[(t - 1, 0)] + (h * dsdt),
                model.s_popf[(t, 0)]
            );
            assert_eq!(
                model.i_popf[(t, 0)],
                model.i_popf[(t - 1, 0)] + (h * didt),
                "Bad i_popf[(t, 0)] at time {}, expected {} got {}",
                t,
                model.i_popf[(t - 1, 0)] + (h * didt),
                model.i_popf[(t, 0)]
            );
            assert_eq!(
                model.r_popf[(t, 0)],
                model.r_popf[(t - 1, 0)] + (h * drdt),
                "Bad r_popf[(t, 0)] at time {}, expected {} got {}",
                t,
                model.r_popf[(t - 1, 0)] + (h * drdt),
                model.r_popf[(t, 0)]
            );
        }
    }

    #[test]
    fn test_init_h() {
        let model: Model = Model {
            length: 10,
            step_size: 1.0,
            i_popf_init: 0.01,
            r_popf_init: 0.0,
            incidence_rate: 0.02,
            removal_rate: 0.03,
            recovery_rate: 0.04,
            s_popf: Mat::new(),
            i_popf: Mat::new(),
            r_popf: Mat::new(),
        };
        let h = model.init_h();
        assert!(h.len() == 4, "Bad h initialization, expected 4 items, got {}", h.len());
        assert!(h[0] == model.step_size / 2.0, "h[0] is not equal to model.step_size/2, got {}", h[0]);
        assert!(h[1] == model.step_size / 2.0, "h[1] is not equal to model.step_size/2, got {}", h[1]);
        assert!(h[2] == model.step_size, "h[2] is not equal to model.step_size, got {}", h[2]);
        assert!(h[3] == model.step_size, "h[3] is not equal to model.step_size, got {}", h[3]);
    }

    #[test]
    fn test_init_y() {
        let model: Model = Model {
            length: 10,
            step_size: 1.0,
            i_popf_init: 0.01,
            r_popf_init: 0.0,
            incidence_rate: 0.02,
            removal_rate: 0.03,
            recovery_rate: 0.04,
            s_popf: Mat::new(),
            i_popf: Mat::new(),
            r_popf: Mat::new(),
        };
        let y = model.init_y();
        assert!(y.len() == 5, "Bad y initialization, expected 5 items, got {}", y.len());
        for i in 0..5 {
            assert!(y[i].s == 0.0, "y[{}].s is not equal to 0.0, got {}", i, y[i].s);
            assert!(y[i].i == 0.0, "y[{}].i is not equal to 0.0, got {}", i, y[i].i);
            assert!(y[i].r == 0.0, "y[{}].r is not equal to 0.0, got {}", i, y[i].r);
        }
    }

    #[test]
    fn test_init_k() {
        let model: Model = Model {
            length: 10,
            step_size: 1.0,
            i_popf_init: 0.01,
            r_popf_init: 0.0,
            incidence_rate: 0.02,
            removal_rate: 0.03,
            recovery_rate: 0.04,
            s_popf: Mat::new(),
            i_popf: Mat::new(),
            r_popf: Mat::new(),
        };
        let k = model.init_k();
        assert!(k.len() == 5, "Bad y initialization, expected 5 items, got {}", k.len());
        for i in 0..5 {
            assert!(k[i].s == 0.0, "k[{}].s is not equal to 0.0, got {}", i, k[i].s);
            assert!(k[i].i == 0.0, "k[{}].i is not equal to 0.0, got {}", i, k[i].i);
            assert!(k[i].r == 0.0, "k[{}].r is not equal to 0.0, got {}", i, k[i].r);
        }
    }

    #[test]
    fn test_run_rk4() {
        let mut model: Model = Model {
            length: 10,
            step_size: 1.0,
            i_popf_init: 0.01,
            r_popf_init: 0.0,
            incidence_rate: 0.02,
            removal_rate: 0.03,
            recovery_rate: 0.04,
            s_popf: Mat::new(),
            i_popf: Mat::new(),
            r_popf: Mat::new(),
        };
        model.init_popf();
        model.run_rk4();
        let h = model.step_size;
        let n = (model.length.to_f64().unwrap() / h)
            .ceil()
            .to_usize()
            .unwrap();
        for t in 0..n - 1 {
            let mut y = model.init_y();
            let mut k = model.init_k();
            let h = model.init_h();
            for i in 0..4 {
                k[i + 1].s = model.dsdt(y[i].s, y[i].i);
                k[i + 1].i = model.didt(y[i].s, y[i].i);
                k[i + 1].r = model.drdt(y[i].i);
                y[i + 1].s = model.next_y(y[0].s, k[i + 1].s, h[i]);
                y[i + 1].i = model.next_y(y[0].i, k[i + 1].i, h[i]);
                y[i + 1].r = model.next_y(y[0].r, k[i + 1].r, h[i]);
            }
            let ds = (k[1].s + (2.0 * k[2].s) + (2.0 * k[3].s) + k[4].s) * (model.step_size / 6.0);
            let di = (k[1].i + (2.0 * k[2].i) + (2.0 * k[3].i) + k[4].i) * (model.step_size / 6.0);
            let dr = (k[1].r + (2.0 * k[2].r) + (2.0 * k[3].r) + k[4].r) * (model.step_size / 6.0);
            model.s_popf[(t + 1, 0)] = model.s_popf[(t, 0)] + ds;
            model.i_popf[(t + 1, 0)] = model.i_popf[(t, 0)] + di;
            model.r_popf[(t + 1, 0)] = model.r_popf[(t, 0)] + dr;
            assert!(
                (model.s_popf[(t, 0)] >= 0.0) & (model.s_popf[(t, 0)] <= 1.0),
                "s_popf[(t, 0)] not in [0, 1] at time {}, got {}",
                t,
                model.s_popf[(t, 0)]
            );
            assert!(
                (model.i_popf[(t, 0)] >= 0.0) & (model.i_popf[(t, 0)] <= 1.0),
                "i_popf[(t, 0)] not in [0, 1] at time {}, got {}",
                t,
                model.i_popf[(t, 0)]
            );
            assert!(
                (model.r_popf[(t, 0)] >= 0.0) & (model.r_popf[(t, 0)] <= 1.0),
                "r_popf[(t, 0)] not in [0, 1] at time {}, got {}",
                t,
                model.r_popf[(t, 0)]
            );
            assert_eq!(
                model.s_popf[(t + 1, 0)],
                model.s_popf[(t, 0)] + ds,
                "Bad s_popf[(t, 0)] at time {}, expected {} got {}",
                t,
                model.s_popf[(t, 0)] + ds,
                model.s_popf[(t + 1, 0)]
            );
            assert_eq!(
                model.i_popf[(t + 1, 0)],
                model.i_popf[(t, 0)] + di,
                "Bad i_popf[(t, 0)] at time {}, expected {} got {}",
                t + 1,
                model.i_popf[(t, 0)] + di,
                model.i_popf[(t + 1, 0)]
            );
            assert_eq!(
                model.r_popf[(t + 1, 0)],
                model.r_popf[(t, 0)] + dr,
                "Bad r_popf[(t, 0)] at time {}, expected {} got {}",
                t + 1,
                model.r_popf[(t, 0)] + dr,
                model.r_popf[(t + 1, 0)]
            );
        }
    }
}
