//! Four compartment DisMod model and methods.
//!
//! Allows transition rates:
//!  - S → C
//!  - S → Ro
//!  - C → S
//!  - C → Rc
//!  - C → Ro
//!
//! See [DisMod's latest documentation](https://dismod-at.readthedocs.io/latest/diff_eq.html#diff-eq-title).
use faer::{Mat, traits::num_traits::ToPrimitive};

/// Numerical integrator variables
///
/// This private struct exists to make indexing k and y during integration
/// simpler.
struct SystemVars {
    s: f64,
    c: f64,
}

/// Create and run a DisMod-type model.
pub struct Model {
    /// Number of indices to generate and solve. The length of the series.
    pub length: usize,
    /// Size of integration step.
    pub step_size: f64,
    /// Initial with-condition population fraction.
    pub c_init: f64,
    /// Transition rate from S into C. Must be in [0, 1].
    pub iota: f64,
    /// Transition rate from C into S. Must be in [0, 1].
    pub rho: f64,
    /// Transition rate from C into Rc. Must be in [0, 1].
    pub chi: f64,
    /// Transition rate from S, C into Ro. Must be in [0, 1].
    pub omega: f64,
    /// Susceptible population fraction at each index. 1D Array with `length` number of elements.
    pub s: Mat<f64>,
    /// With-condition population fraction at each index. 1D Array with `length` number of elements.
    pub c: Mat<f64>,
}

impl Model {
    /// Create an empty model object.
    pub fn new() -> Self {
        return Self {
            length: 0,
            step_size: 0.0,
            c_init: 0.0,
            iota: 0.0,
            rho: 0.0,
            chi: 0.0,
            omega: 0.0,
            s: Mat::new(),
            c: Mat::new(),
        };
    }

    /// Configure model parameters.
    pub fn configure(
        &mut self,
        length: usize,
        step_size: f64,
        c_init: f64,
        iota: f64,
        rho: f64,
        chi: f64,
        omega: f64,
    ) -> &mut Self {
        let n_steps = (length.to_f64().unwrap() / step_size).to_usize().unwrap();
        self.length = length;
        self.step_size = step_size;
        self.c_init = c_init;
        self.iota = iota;
        self.rho = rho;
        self.chi = chi;
        self.omega = omega;
        self.s = Mat::zeros(n_steps, 1);
        self.c = Mat::zeros(n_steps, 1);
        return self;
    }

    /// Initialize population fractions. Creates arrays of length `self.length`
    /// to store the population fractions at each index and sets the 0th index
    /// of each equal to the corresponding initial population fraction.
    pub fn init_popf(&mut self) -> &mut Model {
        let s_init = 1.0 - self.c_init; // Population fractions must sum to 1.
        self.s[(0, 0)] = s_init;
        self.c[(0, 0)] = self.c_init;
        return self;
    }

    fn dsdt(&self, s: f64, c: f64) -> f64 {
        return -((self.iota + self.omega) * s) + (self.rho * c);
    }

    fn dcdt(&self, s: f64, c: f64) -> f64 {
        return (self.iota * s) - ((self.rho + self.chi + self.omega) * c);
    }

    /// Run the DisMod differential equations by the first-order euler method.
    ///
    /// This solution method is very rough and only suitable for demonstration.
    pub fn run_euler(&mut self) -> &Model {
        let h = self.step_size;
        let n = (self.length.to_f64().unwrap() / h)
            .ceil()
            .to_usize()
            .unwrap();
        for t in 1..n - 1 {
            let ds = self.dsdt(self.s[(t, 0)], self.c[(t, 0)]);
            let dc = self.dcdt(self.s[(t, 0)], self.c[(t, 0)]);
            self.s[(t + 1, 0)] = self.s[(t, 0)] + (h * ds);
            self.c[(t + 1, 0)] = self.c[(t, 0)] + (h * dc);
            if t % 10 == 0 {
                println!(
                    "t={:.1} s={:.6} c={:.6}",
                    t.to_f64().unwrap() * self.step_size,
                    self.s[(t, 0)],
                    self.c[(t, 0)],
                );
            }
        }
        return self;
    }

    /// Construct array of runge-kutta intermediate values for each variable.
    fn init_y(&self) -> [SystemVars; 5] {
        return [
            SystemVars { s: 0.0, c: 0.0 },
            SystemVars { s: 0.0, c: 0.0 },
            SystemVars { s: 0.0, c: 0.0 },
            SystemVars { s: 0.0, c: 0.0 },
            SystemVars { s: 0.0, c: 0.0 },
        ];
    }

    /// Construct array of runge-kutta constants for each function.
    fn init_k(&self) -> [SystemVars; 5] {
        return [
            SystemVars { s: 0.0, c: 0.0 },
            SystemVars { s: 0.0, c: 0.0 },
            SystemVars { s: 0.0, c: 0.0 },
            SystemVars { s: 0.0, c: 0.0 },
            SystemVars { s: 0.0, c: 0.0 },
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
        y[0].s = self.s[(t, 0)];
        y[0].c = self.c[(t, 0)];
        for i in 0..4 {
            k[i + 1].s = self.dsdt(y[i].s, y[i].c);
            k[i + 1].c = self.dcdt(y[i].s, y[i].c);
            y[i + 1].s = self.next_y(y[0].s, k[i + 1].s, h[i]);
            y[i + 1].c = self.next_y(y[0].c, k[i + 1].c, h[i]);
        }
        return k;
    }

    /// Run the DisMod differential equations by the 4th order Runge-Kutta method.
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
            let dc = (k[1].c + (2.0 * k[2].c) + (2.0 * k[3].c) + k[4].c) * (self.step_size / 6.0);
            self.s[(t + 1, 0)] = self.s[(t, 0)] + ds;
            self.c[(t + 1, 0)] = self.c[(t, 0)] + dc;
            if t % 10 == 0 {
                println!(
                    "t={:.1} s={:.6} c={:.6}",
                    t.to_f64().unwrap() * self.step_size,
                    self.s[(t, 0)],
                    self.c[(t, 0)],
                );
            }
        }
        return self;
    }
}

#[cfg(test)]
mod tests {
    use crate::sirrs::dismod::Model;
    use faer::{Mat, traits::num_traits::ToPrimitive};

    #[test]
    fn test_new() {
        let model = Model::new();
        assert_eq!(
            model.length, 0,
            "Bad length, expected 0 got {}",
            model.length
        );
        assert_eq!(
            model.c_init, 0.0,
            "Bad c_init, expected 0.0 got {}",
            model.c_init,
        );
        assert_eq!(model.iota, 0.0, "Bad iota, expected 0.0 got {}", model.iota,);
        assert_eq!(model.rho, 0.0, "Bad rho, expected 0.0 got {}", model.rho);
        assert_eq!(model.chi, 0.0, "Bad chi, expected 0.0 got {}", model.chi);
        assert_eq!(
            model.omega, 0.0,
            "Bad omega, expected 0.0 got {}",
            model.omega
        );
        assert_eq!(
            model.s,
            Mat::new(),
            "Bad s, expected Mat::new() got {:?}",
            model.s,
        );
        assert_eq!(
            model.c,
            Mat::new(),
            "Bad c, expected Mat::new() got {:?}",
            model.c,
        );
    }

    #[test]
    fn test_configure() {
        let mut model = Model::new();
        model.configure(10, 1.0, 0.01, 0.01, 0.02, 0.03, 0.04);
        let n_steps = (model.length.to_f64().unwrap() / model.step_size)
            .to_usize()
            .unwrap();
        assert_eq!(
            model.length, 10,
            "Bad length, expected 10 got {}",
            model.length
        );
        assert_eq!(
            model.c_init, 0.01,
            "Bad c_init, expected 0.01 got {}",
            model.c_init,
        );
        assert_eq!(
            model.iota, 0.01,
            "Bad iota, expected 0.01 got {}",
            model.iota,
        );
        assert_eq!(model.rho, 0.02, "Bad rho, expected 0.02 got {}", model.rho);
        assert_eq!(model.chi, 0.03, "Bad chi, expected 0.03 got {}", model.chi);
        assert_eq!(
            model.omega, 0.04,
            "Bad omega, expected 0.04 got {}",
            model.omega
        );
        assert_eq!(
            model.s,
            Mat::zeros(n_steps, 1),
            "Bad s, expected Mat::zeros(n_steps, 1) got {:?}",
            model.s,
        );
        assert_eq!(
            model.c,
            Mat::zeros(n_steps, 1),
            "Bad c, expected Mat::zeros(n_steps, 1) got {:?}",
            model.c,
        );
    }

    #[test]
    fn test_init_popf() {
        let mut model = Model::new();
        model.configure(10, 1.0, 0.01, 0.01, 0.02, 0.03, 0.04);
        model.init_popf();
        assert_eq!(
            model.s.shape(),
            (model.length, 1),
            "Bad s dimensions, expected {:?} got {:?}.",
            (model.length, 1),
            model.s.shape(),
        );
        assert_eq!(
            model.c.shape(),
            (model.length, 1),
            "Bad c dimensions, expected {:?} got {:?}.",
            (model.length, 1),
            model.c.shape(),
        );
        assert_eq!(
            model.s[(0, 0)],
            1.0 - model.c_init,
            "Bad s[(0, 0)] initialization value, expected {} got {}.",
            1.0 - model.c_init,
            model.s[(0, 0)],
        );
        assert_eq!(
            model.c[(0, 0)],
            model.c_init,
            "Bad c[(0, 0)] initialization value, expected {} got {}.",
            model.c_init,
            model.c[(0, 0)],
        );
        for t in 1..model.length {
            assert_eq!(
                model.s[(t, 0)],
                0.0,
                "Bad s[t>0] initialization value, expected 0.0 got {}.",
                model.s[(t, 0)]
            );
            assert_eq!(
                model.c[(t, 0)],
                0.0,
                "Bad c[t>0] initialization value, expected 0.0 got {}.",
                model.c[(t, 0)]
            );
        }
    }

    #[test]
    fn test_run_euler() {
        let mut model = Model::new();
        model.configure(10, 1.0, 0.01, 0.01, 0.02, 0.03, 0.04);
        model.init_popf();
        model.run_euler();
        for t in 1..model.length {
            let dsdt = -((model.iota + model.omega) * model.s[(t - 1, 0)])
                + (model.rho * model.c[(t - 1, 0)]);
            let dcdt = (model.iota * model.s[(t - 1, 0)])
                - ((model.rho + model.chi + model.omega) * model.c[(t - 1, 0)]);
            model.s[(t, 0)] = model.s[(t - 1, 0)] + dsdt;
            model.c[(t, 0)] = model.c[(t - 1, 0)] + dcdt;
            assert!(
                (model.s[(t, 0)] >= 0.0) & (model.s[(t, 0)] <= 1.0),
                "s[(t, 0)] not in [0, 1] at time {}, got {}",
                t,
                model.s[(t, 0)]
            );
            assert!(
                (model.c[(t, 0)] >= 0.0) & (model.c[(t, 0)] <= 1.0),
                "c[(t, 0)] not in [0, 1] at time {}, got {}",
                t,
                model.c[(t, 0)]
            );
            assert_eq!(
                model.s[(t, 0)],
                model.s[(t - 1, 0)] + dsdt,
                "Bad s[(t, 0)] at time {}, expected {} got {}",
                t,
                model.s[(t - 1, 0)] + dsdt,
                model.s[(t, 0)]
            );
            assert_eq!(
                model.c[(t, 0)],
                model.c[(t - 1, 0)] + dcdt,
                "Bad c[(t, 0)] at time {}, expected {} got {}",
                t,
                model.c[(t - 1, 0)] + dcdt,
                model.c[(t, 0)]
            );
        }
    }

    #[test]
    fn test_init_h() {
        let mut model = Model::new();
        model.configure(10, 1.0, 0.01, 0.01, 0.02, 0.03, 0.04);
        let h = model.init_h();
        assert!(
            h.len() == 4,
            "Bad h initialization, expected 4 items, got {}",
            h.len()
        );
        assert!(
            h[0] == model.step_size / 2.0,
            "h[0] is not equal to model.step_size/2, got {}",
            h[0]
        );
        assert!(
            h[1] == model.step_size / 2.0,
            "h[1] is not equal to model.step_size/2, got {}",
            h[1]
        );
        assert!(
            h[2] == model.step_size,
            "h[2] is not equal to model.step_size, got {}",
            h[2]
        );
        assert!(
            h[3] == model.step_size,
            "h[3] is not equal to model.step_size, got {}",
            h[3]
        );
    }

    #[test]
    fn test_init_y() {
        let mut model = Model::new();
        model.configure(10, 1.0, 0.01, 0.01, 0.02, 0.03, 0.04);
        let y = model.init_y();
        assert!(
            y.len() == 5,
            "Bad y initialization, expected 5 items, got {}",
            y.len()
        );
        for i in 0..5 {
            assert!(
                y[i].s == 0.0,
                "y[{}].s is not equal to 0.0, got {}",
                i,
                y[i].s
            );
            assert!(
                y[i].c == 0.0,
                "y[{}].c is not equal to 0.0, got {}",
                i,
                y[i].c
            );
        }
    }

    #[test]
    fn test_init_k() {
        let mut model = Model::new();
        model.configure(10, 1.0, 0.01, 0.01, 0.02, 0.03, 0.04);
        let k = model.init_k();
        assert!(
            k.len() == 5,
            "Bad y initialization, expected 5 items, got {}",
            k.len()
        );
        for i in 0..5 {
            assert!(
                k[i].s == 0.0,
                "k[{}].s is not equal to 0.0, got {}",
                i,
                k[i].s
            );
            assert!(
                k[i].c == 0.0,
                "k[{}].c is not equal to 0.0, got {}",
                i,
                k[i].c
            );
        }
    }

    #[test]
    fn test_run_rk4() {
        let mut model = Model::new();
        model.configure(10, 1.0, 0.01, 0.01, 0.02, 0.03, 0.04);
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
                k[i + 1].s = model.dsdt(y[i].s, y[i].s);
                k[i + 1].c = model.dcdt(y[i].s, y[i].c);
                y[i + 1].s = model.next_y(y[0].s, k[i + 1].s, h[i]);
                y[i + 1].c = model.next_y(y[0].c, k[i + 1].c, h[i]);
            }
            let ds = (k[1].s + (2.0 * k[2].s) + (2.0 * k[3].s) + k[4].s) * (model.step_size / 6.0);
            let di = (k[1].c + (2.0 * k[2].c) + (2.0 * k[3].c) + k[4].c) * (model.step_size / 6.0);
            model.s[(t + 1, 0)] = model.s[(t, 0)] + ds;
            model.c[(t + 1, 0)] = model.c[(t, 0)] + di;
            assert!(
                (model.s[(t, 0)] >= 0.0) & (model.s[(t, 0)] <= 1.0),
                "s_popf[(t, 0)] not in [0, 1] at time {}, got {}",
                t,
                model.s[(t, 0)]
            );
            assert!(
                (model.c[(t, 0)] >= 0.0) & (model.c[(t, 0)] <= 1.0),
                "i_popf[(t, 0)] not in [0, 1] at time {}, got {}",
                t,
                model.c[(t, 0)]
            );
            assert_eq!(
                model.s[(t + 1, 0)],
                model.s[(t, 0)] + ds,
                "Bad s_popf[(t, 0)] at time {}, expected {} got {}",
                t,
                model.s[(t, 0)] + ds,
                model.s[(t + 1, 0)]
            );
            assert_eq!(
                model.c[(t + 1, 0)],
                model.c[(t, 0)] + di,
                "Bad i_popf[(t, 0)] at time {}, expected {} got {}",
                t + 1,
                model.c[(t, 0)] + di,
                model.c[(t + 1, 0)]
            );
        }
    }
}
