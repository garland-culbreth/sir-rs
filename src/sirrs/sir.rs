//! Three compartment SIR model and methods.
//!
//! Allows transition rates:
//!  - S → I  
//!  - I → R  
//!  - R → S  
use faer::{Mat, traits::num_traits::ToPrimitive};

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

    fn ds(&self, susceptible: f64, infectious: f64) -> f64 {
        return (-self.incidence_rate * susceptible * infectious)
            + (self.recovery_rate * infectious);
    }

    fn di(&self, susceptible: f64, infectious: f64) -> f64 {
        return (self.incidence_rate * susceptible * infectious)
            - ((self.recovery_rate + self.removal_rate) * infectious);
    }

    fn dr(&self, infectious: f64) -> f64 {
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
            let dsdt = self.ds(self.s_popf[(i, 0)], self.i_popf[(i, 0)]);
            let didt = self.di(self.s_popf[(i, 0)], self.i_popf[(i, 0)]);
            let drdt = self.dr(self.i_popf[(i, 0)]);
            self.s_popf[(i + 1, 0)] = self.s_popf[(i, 0)] + (h * dsdt);
            self.i_popf[(i + 1, 0)] = self.i_popf[(i, 0)] + (h * didt);
            self.r_popf[(i + 1, 0)] = self.r_popf[(i, 0)] + (h * drdt);
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

    /// Solve the system by the 4th order Runge-Kutta method.
    ///
    /// This method is suitable for general purposes.
    pub fn run_rk4(&mut self) -> &Model {
        let h = self.step_size;
        let n = (self.length.to_f64().unwrap() / h)
            .ceil()
            .to_usize()
            .unwrap();
        for i in 0..n - 1 {
            //
            let y0 = [
                self.s_popf[(i, 0)],
                self.i_popf[(i, 0)],
                self.r_popf[(i, 0)],
            ];
            let k1 = [self.ds(y0[0], y0[1]), self.di(y0[0], y0[1]), self.dr(y0[1])];
            let y1 = [
                y0[0] + (k1[0] * (h / 2.0)),
                y0[1] + (k1[1] * (h / 2.0)),
                y0[2] + (k1[2] * (h / 2.0)),
            ];
            let k2 = [self.ds(y1[0], y1[1]), self.di(y1[0], y1[1]), self.dr(y1[1])];
            let y2 = [
                y0[0] + (k2[0] * (h / 2.0)),
                y0[1] + (k2[1] * (h / 2.0)),
                y0[2] + (k2[2] * (h / 2.0)),
            ];
            let k3 = [self.ds(y2[0], y2[1]), self.di(y2[0], y2[1]), self.dr(y2[1])];
            let y3 = [
                y0[0] + (k3[0] * h),
                y0[1] + (k3[1] * h),
                y0[2] + (k3[2] * h),
            ];
            let k4 = [self.ds(y3[0], y3[1]), self.di(y3[0], y3[1]), self.dr(y3[1])];
            let dsdt = (k1[0] + (2.0 * k2[0]) + (2.0 * k3[0]) + k4[0]) / 6.0;
            let didt = (k1[1] + (2.0 * k2[1]) + (2.0 * k3[1]) + k4[1]) / 6.0;
            let drdt = (k1[2] + (2.0 * k2[2]) + (2.0 * k3[2]) + k4[2]) / 6.0;
            self.s_popf[(i + 1, 0)] = self.s_popf[(i, 0)] + dsdt;
            self.i_popf[(i + 1, 0)] = self.i_popf[(i, 0)] + didt;
            self.r_popf[(i + 1, 0)] = self.r_popf[(i, 0)] + drdt;
            println!(
                "t={}: s={:.6} i={:.6} r={:.6}",
                i,
                self.s_popf[(i, 0)],
                self.i_popf[(i, 0)],
                self.r_popf[(i, 0)],
            );
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
            let dsdt = model.ds(model.s_popf[(t - 1, 0)], model.i_popf[(t - 1, 0)]);
            let didt = model.di(model.s_popf[(t - 1, 0)], model.i_popf[(t - 1, 0)]);
            let drdt = model.dr(model.i_popf[(t - 1, 0)]);
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
}
