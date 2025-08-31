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
    /// Initialize population fractions. Creates arrays of length `self.length`
    /// to store the population fractions at each index and sets the 0th index
    /// of each equal to the corresponding initial population fraction.
    pub fn init_popf(&mut self) -> &mut Model {
        let n_steps = (self.length.to_f64().unwrap() / self.step_size)
            .to_usize()
            .unwrap();
        self.s = Mat::zeros(n_steps, 1);
        self.c = Mat::zeros(n_steps, 1);
        let s_init = 1.0 - self.c_init; // Population fractions must sum to 1.
        self.s[(0, 0)] = s_init;
        self.c[(0, 0)] = self.c_init;
        return self;
    }

    fn ds(&self, s: f64, c: f64) -> f64 {
        return -((self.iota + self.omega) * s) + (self.rho * c);
    }

    fn dc(&self, s: f64, c: f64) -> f64 {
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
            let dsdt = self.ds(self.s[(t, 0)], self.c[(t, 0)]);
            let dcdt = self.dc(self.s[(t, 0)], self.c[(t, 0)]);
            self.s[(t + 1, 0)] = self.s[(t, 0)] + (h * dsdt);
            self.c[(t + 1, 0)] = self.c[(t, 0)] + (h * dcdt);
            println!("t={}: s={:.6} c={:.6}", t, self.s[(t, 0)], self.c[(t, 0)],);
        }
        return self;
    }

    /// Run the DisMod differential equations by the 4th order Runge-Kutta method.
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
            let y0 = [self.s[(i, 0)], self.c[(i, 0)]];
            let k1 = [self.ds(y0[0], y0[1]), self.dc(y0[0], y0[1])];
            let y1 = [y0[0] + (k1[0] * (h / 2.0)), y0[1] + (k1[1] * (h / 2.0))];
            let k2 = [self.ds(y1[0], y1[1]), self.dc(y1[0], y1[1])];
            let y2 = [y0[0] + (k2[0] * (h / 2.0)), y0[1] + (k2[1] * (h / 2.0))];
            let k3 = [self.ds(y2[0], y2[1]), self.dc(y2[0], y2[1])];
            let y3 = [y0[0] + (k3[0] * (h)), y0[1] + (k3[1] * h)];
            let k4 = [self.ds(y3[0], y3[1]), self.dc(y3[0], y3[1])];
            let dsdt = (k1[0] + (2.0 * k2[0]) + (2.0 * k3[0]) + k4[0]) / 6.0;
            let dcdt = (k1[1] + (2.0 * k2[1]) + (2.0 * k3[1]) + k4[1]) / 6.0;
            self.s[(i + 1, 0)] = self.s[(i, 0)] + dsdt;
            self.c[(i + 1, 0)] = self.c[(i, 0)] + dcdt;
            println!("t={}: s={:.6} c={:.6}", i, self.s[(i, 0)], self.c[(i, 0)],);
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
            step_size: 1.0,
            c_init: 0.01,
            iota: 0.01,
            rho: 0.02,
            chi: 0.03,
            omega: 0.04,
            s: Mat::new(),
            c: Mat::new(),
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
    }

    #[test]
    fn test_init_popf() {
        let mut model: Model = Model {
            length: 10,
            step_size: 1.0,
            c_init: 0.01,
            iota: 0.0,
            rho: 0.02,
            chi: 0.03,
            omega: 0.04,
            s: Mat::new(),
            c: Mat::new(),
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
        let mut model: Model = Model {
            length: 10,
            step_size: 1.0,
            c_init: 0.01,
            iota: 0.0,
            rho: 0.02,
            chi: 0.03,
            omega: 0.04,
            s: Mat::new(),
            c: Mat::new(),
        };
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
}
