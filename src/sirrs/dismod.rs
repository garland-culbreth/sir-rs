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
use nalgebra::DVector;

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
    pub s: DVector<f64>,
    /// With-condition population fraction at each index. 1D Array with `length` number of elements.
    pub c: DVector<f64>,
    /// Removed by condition population fraction at each index. 1D Array with `length` number of elements.
    pub rc: DVector<f64>,
    /// Removed by other population fraction at each index. 1D Array with `length` number of elements.
    pub ro: DVector<f64>,
}

impl Model {
    /// Initialize population fractions. Creates arrays of length `self.length`
    /// to store the population fractions at each index and sets the 0th index
    /// of each equal to the corresponding initial population fraction.
    pub fn init_popf(&mut self) -> &mut Model {
        self.s = DVector::<f64>::zeros(self.length);
        self.c = DVector::<f64>::zeros(self.length);
        self.rc = DVector::<f64>::zeros(self.length);
        self.ro = DVector::<f64>::zeros(self.length);
        let s_init = 1.0 - self.c_init; // Population fractions must sum to 1.
        self.s[0] = s_init;
        self.c[0] = self.c_init;
        return self;
    }

    /// Run the DisMod differential equations by the first-order finite difference
    /// method.
    ///
    /// This solution method is very rough and only suitable for demonstration.
    pub fn run_fdm_o1(&mut self) -> &Model {
        for t in 1..self.s.len() {
            let dsdt = -((self.iota + self.omega) * self.s[t - 1]) + (self.rho * self.c[t - 1]);
            let dcdt =
                (self.iota * self.s[t - 1]) - ((self.rho + self.chi + self.omega) * self.c[t - 1]);
            let drcdt = self.chi * self.c[t - 1];
            let drodt = self.omega * (self.s[t - 1] + self.c[t - 1]);
            self.s[t] = self.s[t - 1] + dsdt;
            self.c[t] = self.c[t - 1] + dcdt;
            self.rc[t] = self.rc[t - 1] + drcdt;
            self.ro[t] = self.ro[t - 1] + drodt;
            println!(
                "t={}: s={:.6} c={:.6} rc={:.6} ro={:.6}",
                t, self.s[t], self.c[t], self.rc[t], self.ro[t]
            );
        }
        return self;
    }
}

#[cfg(test)]
mod tests {
    use crate::sirrs::dismod::Model;
    use nalgebra::DVector;

    #[test]
    fn test_init_model() {
        let model: Model = Model {
            length: 10,
            c_init: 0.01,
            iota: 0.01,
            rho: 0.02,
            chi: 0.03,
            omega: 0.04,
            s: DVector::default(),
            c: DVector::default(),
            ro: DVector::default(),
            rc: DVector::default(),
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
            DVector::default(),
            "Bad s, expected DVector::default() got {}",
            model.s,
        );
        assert_eq!(
            model.c,
            DVector::default(),
            "Bad c, expected DVector::default() got {}",
            model.c,
        );
        assert_eq!(
            model.rc,
            DVector::default(),
            "Bad rc, expected DVector::default() got {}",
            model.rc,
        );
        assert_eq!(
            model.ro,
            DVector::default(),
            "Bad ro, expected DVector::default() got {}",
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
            s: DVector::default(),
            c: DVector::default(),
            ro: DVector::default(),
            rc: DVector::default(),
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
            model.s[0],
            1.0 - model.c_init,
            "Bad s[0] initialization value, expected {} got {}.",
            1.0 - model.c_init,
            model.s[0],
        );
        assert_eq!(
            model.c[0], model.c_init,
            "Bad c[0] initialization value, expected {} got {}.",
            model.c_init, model.c[0],
        );
        assert_eq!(
            model.ro[0], 0.0,
            "Bad ro[0] initialization value, expected {} got {}.",
            0.0, model.c[0],
        );
        assert_eq!(
            model.rc[0], 0.0,
            "Bad rc[0] initialization value, expected {} got {}.",
            0.0, model.c[0],
        );
        for t in 1..model.length {
            assert_eq!(
                model.s[t], 0.0,
                "Bad s[t>0] initialization value, expected 0.0 got {}.",
                model.s[t]
            );
            assert_eq!(
                model.c[t], 0.0,
                "Bad c[t>0] initialization value, expected 0.0 got {}.",
                model.c[t]
            );
            assert_eq!(
                model.rc[t], 0.0,
                "Bad rc[t>0] initialization value, expected 0.0 got {}.",
                model.rc[t]
            );
            assert_eq!(
                model.ro[t], 0.0,
                "Bad ro[t>0] initialization value, expected 0.0 got {}.",
                model.ro[t]
            );
        }
    }

    #[test]
    fn test_run_fdm_o1() {
        let mut model: Model = Model {
            length: 10,
            c_init: 0.01,
            iota: 0.0,
            rho: 0.02,
            chi: 0.03,
            omega: 0.04,
            s: DVector::default(),
            c: DVector::default(),
            ro: DVector::default(),
            rc: DVector::default(),
        };
        model.init_popf();
        model.run_fdm_o1();
        for t in 1..model.length {
            let dsdt =
                -((model.iota + model.omega) * model.s[t - 1]) + (model.rho * model.c[t - 1]);
            let dcdt = (model.iota * model.s[t - 1])
                - ((model.rho + model.chi + model.omega) * model.c[t - 1]);
            let drcdt = model.chi * model.c[t - 1];
            let drodt = model.omega * (model.s[t - 1] + model.c[t - 1]);
            model.s[t] = model.s[t - 1] + dsdt;
            model.c[t] = model.c[t - 1] + dcdt;
            model.rc[t] = model.rc[t - 1] + drcdt;
            model.ro[t] = model.ro[t - 1] + drodt;
            assert!(
                (model.s[t] >= 0.0) & (model.s[t] <= 1.0),
                "s[t] not in [0, 1] at time {}, got {}",
                t,
                model.s[t]
            );
            assert!(
                (model.c[t] >= 0.0) & (model.c[t] <= 1.0),
                "c[t] not in [0, 1] at time {}, got {}",
                t,
                model.c[t]
            );
            assert!(
                (model.rc[t] >= 0.0) & (model.rc[t] <= 1.0),
                "rc[t] not in [0, 1] at time {}, got {}",
                t,
                model.rc[t]
            );
            assert!(
                (model.ro[t] >= 0.0) & (model.ro[t] <= 1.0),
                "ro[t] not in [0, 1] at time {}, got {}",
                t,
                model.ro[t]
            );
            assert_eq!(
                model.s[t],
                model.s[t - 1] + dsdt,
                "Bad s[t] at time {}, expected {} got {}",
                t,
                model.s[t - 1] + dsdt,
                model.s[t]
            );
            assert_eq!(
                model.c[t],
                model.c[t - 1] + dcdt,
                "Bad c[t] at time {}, expected {} got {}",
                t,
                model.c[t - 1] + dcdt,
                model.c[t]
            );
            assert_eq!(
                model.rc[t],
                model.rc[t - 1] + drcdt,
                "Bad rc[t] at time {}, expected {} got {}",
                t,
                model.rc[t - 1] + drcdt,
                model.rc[t]
            );
            assert_eq!(
                model.ro[t],
                model.ro[t - 1] + drodt,
                "Bad ro[t] at time {}, expected {} got {}",
                t,
                model.ro[t - 1] + drodt,
                model.ro[t]
            );
        }
    }
}
