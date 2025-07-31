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
use faer::Mat;

/// Create and run a DisMod-type model.
pub struct Model {
    /// Number of indices to generate and solve. The length of the series.
    pub length: usize,
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
    /// Removed by condition population fraction at each index. 1D Array with `length` number of elements.
    pub rc: Mat<f64>,
    /// Removed by other population fraction at each index. 1D Array with `length` number of elements.
    pub ro: Mat<f64>,
}

impl Model {
    /// Initialize population fractions. Creates arrays of length `self.length`
    /// to store the population fractions at each index and sets the 0th index
    /// of each equal to the corresponding initial population fraction.
    pub fn init_popf(&mut self) -> &mut Model {
        self.s = Mat::zeros(self.length, 1);
        self.c = Mat::zeros(self.length, 1);
        self.rc = Mat::zeros(self.length, 1);
        self.ro = Mat::zeros(self.length, 1);
        let s_init = 1.0 - self.c_init; // Population fractions must sum to 1.
        self.s[(0, 0)] = s_init;
        self.c[(0, 0)] = self.c_init;
        return self;
    }

    /// Run the DisMod differential equations by the first-order euler method.
    ///
    /// This solution method is very rough and only suitable for demonstration.
    pub fn run_euler(&mut self) -> &Model {
        for t in 1..self.length {
            let dsdt = -((self.iota + self.omega) * self.s[(t - 1, 0)]) + (self.rho * self.c[(t - 1, 0)]);
            let dcdt =
                (self.iota * self.s[(t - 1, 0)]) - ((self.rho + self.chi + self.omega) * self.c[(t - 1, 0)]);
            let drcdt = self.chi * self.c[(t - 1, 0)];
            let drodt = self.omega * (self.s[(t - 1, 0)] + self.c[(t - 1, 0)]);
            self.s[(t, 0)] = self.s[(t - 1, 0)] + dsdt;
            self.c[(t, 0)] = self.c[(t - 1, 0)] + dcdt;
            self.rc[(t, 0)] = self.rc[(t - 1, 0)] + drcdt;
            self.ro[(t, 0)] = self.ro[(t - 1, 0)] + drodt;
            println!(
                "t={}: s={:.6} c={:.6} rc={:.6} ro={:.6}",
                t, self.s[(t, 0)], self.c[(t, 0)], self.rc[(t, 0)], self.ro[(t, 0)]
            );
        }
        return self;
    }
    
    fn ds(&self, susceptible: f64, condition: f64) -> f64 {
        return -((self.iota + self.omega) * susceptible) + (self.rho * condition);
    }

    fn dc(&self, susceptible: f64, condition: f64) -> f64 {
        return (self.iota * susceptible) - ((self.rho + self.chi + self.omega) * condition);
    }

    fn rk4_step<F>(&self, f: F, x: f64, y: f64, step_size: f64) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let k1 = step_size * f(x, y);
        let k2 = step_size * f(x + (step_size / 2.0), y + (k1 / 2.0));
        let k3 = step_size * f(x + (step_size / 2.0), y + (k2 / 2.0));
        let k4 = step_size * f(x + step_size, y + k3);
        let df = (k1 + (2.0 * k2) + (2.0 * k3) + k4) / 6.0;
        return df;
    }

    /// Solve the system by the 4th order Runge-Kutta method.
    ///
    /// This method is suitable for general purposes.
    pub fn run_rk4(&mut self) -> &Model {
        let step_size = 0.01;
        for t in 0..self.length - 1 {
            let s_t = self.s[(t, 0)];
            let c_t = self.c[(t, 0)];
            let dsdt = self.rk4_step(|x, y| self.ds(x, y), s_t, c_t, step_size);
            let dcdt = self.rk4_step(|x, y| self.dc(x, y), s_t, c_t, step_size);
            self.s[(t + 1, 0)] = self.s[(t, 0)] + dsdt;
            self.c[(t + 1, 0)] = self.c[(t, 0)] + dcdt;
            println!(
                "t={}: s={:.6} c={:.6}",
                t,
                self.s[(t, 0)],
                self.c[(t, 0)],
            );
        }
        return self;
    }
}

#[cfg(test)]
mod tests {
    use crate::sirrs::dismod::Model;
    use faer::Mat;

    #[test]
    fn test_init_model() {
        let model: Model = Model {
            length: 10,
            c_init: 0.01,
            iota: 0.01,
            rho: 0.02,
            chi: 0.03,
            omega: 0.04,
            s: Mat::new(),
            c: Mat::new(),
            ro: Mat::new(),
            rc: Mat::new(),
        };
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
        assert_eq!(
            model.rc,
            Mat::new(),
            "Bad rc, expected Mat::new() got {:?}",
            model.rc,
        );
        assert_eq!(
            model.ro,
            Mat::new(),
            "Bad ro, expected Mat::new() got {:?}",
            model.ro,
        );
    }

    #[test]
    fn test_init_popf() {
        let mut model: Model = Model {
            length: 10,
            c_init: 0.01,
            iota: 0.0,
            rho: 0.02,
            chi: 0.03,
            omega: 0.04,
            s: Mat::new(),
            c: Mat::new(),
            ro: Mat::new(),
            rc: Mat::new(),
        };
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
            model.rc.shape(),
            (model.length, 1),
            "Bad rc dimensions, expected {:?} got {:?}.",
            (model.length, 1),
            model.rc.shape(),
        );
        assert_eq!(
            model.ro.shape(),
            (model.length, 1),
            "Bad ro dimensions, expected {:?} got {:?}.",
            (model.length, 1),
            model.ro.shape(),
        );
        assert_eq!(
            model.s[(0, 0)],
            1.0 - model.c_init,
            "Bad s[(0, 0)] initialization value, expected {} got {}.",
            1.0 - model.c_init,
            model.s[(0, 0)],
        );
        assert_eq!(
            model.c[(0, 0)], model.c_init,
            "Bad c[(0, 0)] initialization value, expected {} got {}.",
            model.c_init, model.c[(0, 0)],
        );
        assert_eq!(
            model.ro[(0, 0)], 0.0,
            "Bad ro[(0, 0)] initialization value, expected {} got {}.",
            0.0, model.c[(0, 0)],
        );
        assert_eq!(
            model.rc[(0, 0)], 0.0,
            "Bad rc[(0, 0)] initialization value, expected {} got {}.",
            0.0, model.c[(0, 0)],
        );
        for t in 1..model.length {
            assert_eq!(
                model.s[(t, 0)], 0.0,
                "Bad s[t>0] initialization value, expected 0.0 got {}.",
                model.s[(t, 0)]
            );
            assert_eq!(
                model.c[(t, 0)], 0.0,
                "Bad c[t>0] initialization value, expected 0.0 got {}.",
                model.c[(t, 0)]
            );
            assert_eq!(
                model.rc[(t, 0)], 0.0,
                "Bad rc[t>0] initialization value, expected 0.0 got {}.",
                model.rc[(t, 0)]
            );
            assert_eq!(
                model.ro[(t, 0)], 0.0,
                "Bad ro[t>0] initialization value, expected 0.0 got {}.",
                model.ro[(t, 0)]
            );
        }
    }

    #[test]
    fn test_run_euler() {
        let mut model: Model = Model {
            length: 10,
            c_init: 0.01,
            iota: 0.0,
            rho: 0.02,
            chi: 0.03,
            omega: 0.04,
            s: Mat::new(),
            c: Mat::new(),
            ro: Mat::new(),
            rc: Mat::new(),
        };
        model.init_popf();
        model.run_euler();
        for t in 1..model.length {
            let dsdt =
                -((model.iota + model.omega) * model.s[(t - 1, 0)]) + (model.rho * model.c[(t - 1, 0)]);
            let dcdt = (model.iota * model.s[(t - 1, 0)])
                - ((model.rho + model.chi + model.omega) * model.c[(t - 1, 0)]);
            let drcdt = model.chi * model.c[(t - 1, 0)];
            let drodt = model.omega * (model.s[(t - 1, 0)] + model.c[(t - 1, 0)]);
            model.s[(t, 0)] = model.s[(t - 1, 0)] + dsdt;
            model.c[(t, 0)] = model.c[(t - 1, 0)] + dcdt;
            model.rc[(t, 0)] = model.rc[(t - 1, 0)] + drcdt;
            model.ro[(t, 0)] = model.ro[(t - 1, 0)] + drodt;
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
            assert!(
                (model.rc[(t, 0)] >= 0.0) & (model.rc[(t, 0)] <= 1.0),
                "rc[(t, 0)] not in [0, 1] at time {}, got {}",
                t,
                model.rc[(t, 0)]
            );
            assert!(
                (model.ro[(t, 0)] >= 0.0) & (model.ro[(t, 0)] <= 1.0),
                "ro[(t, 0)] not in [0, 1] at time {}, got {}",
                t,
                model.ro[(t, 0)]
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
            assert_eq!(
                model.rc[(t, 0)],
                model.rc[(t - 1, 0)] + drcdt,
                "Bad rc[(t, 0)] at time {}, expected {} got {}",
                t,
                model.rc[(t - 1, 0)] + drcdt,
                model.rc[(t, 0)]
            );
            assert_eq!(
                model.ro[(t, 0)],
                model.ro[(t - 1, 0)] + drodt,
                "Bad ro[(t, 0)] at time {}, expected {} got {}",
                t,
                model.ro[(t - 1, 0)] + drodt,
                model.ro[(t, 0)]
            );
        }
    }
}
