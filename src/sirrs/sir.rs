//! Three compartment SIR model and methods.
//!
//! Allows transition rates:
//!  - S → I  
//!  - I → R  
//!  - R → S  
use nalgebra::DVector;

/// Create and run an SIR model.
pub struct Model {
    /// Number of indices to generate and solve. The length of the series.
    pub length: usize,
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
    pub s_popf: DVector<f64>,
    /// Inectious population fraction at each index. 1D Array with `length` number of elements.
    pub i_popf: DVector<f64>,
    /// Removed population fraction at each index. 1D Array with `length` number of elements.
    pub r_popf: DVector<f64>,
}

impl Model {
    /// Initialize population fractions. Creates arrays of length `self.length`
    /// to store the population fractions at each index and sets the 0th index
    /// of each equal to the corresponding initial population fraction.
    pub fn init_popf(&mut self) -> &mut Model {
        self.s_popf = DVector::<f64>::zeros(self.length);
        self.i_popf = DVector::<f64>::zeros(self.length);
        self.r_popf = DVector::<f64>::zeros(self.length);
        let s_init = 1.0 - self.i_popf_init - self.r_popf_init; // Population fractions must sum to 1.
        self.s_popf[0] = s_init;
        self.i_popf[0] = self.i_popf_init;
        self.r_popf[0] = self.r_popf_init;
        return self;
    }

    /// Run the SIR differential equations by the first-order finite difference
    /// method.
    ///
    /// This solution method is very rough and only suitable for demonstration.
    pub fn run_fdm_o1(&mut self) -> &Model {
        for t in 1..self.s_popf.len() {
            let dsdt = (-self.incidence_rate * self.s_popf[t - 1] * self.i_popf[t - 1])
                + (self.recovery_rate * self.i_popf[t - 1]);
            let didt = (self.incidence_rate * self.s_popf[t - 1] * self.i_popf[t - 1])
                - (self.removal_rate * self.i_popf[t - 1])
                - (self.recovery_rate * self.i_popf[t - 1]);
            let drdt = self.removal_rate * self.i_popf[t - 1];
            self.s_popf[t] = self.s_popf[t - 1] + dsdt;
            self.i_popf[t] = self.i_popf[t - 1] + didt;
            self.r_popf[t] = self.r_popf[t - 1] + drdt;
            println!(
                "t={}: s={:.6} i={:.6} r={:.6}",
                t, self.s_popf[t], self.i_popf[t], self.r_popf[t]
            );
        }
        return self;
    }
}

#[cfg(test)]
mod tests {
    use crate::sirrs::sir::Model;
    use nalgebra::DVector;

    #[test]
    fn test_init_model() {
        let model: Model = Model {
            length: 10,
            i_popf_init: 0.01,
            r_popf_init: 0.0,
            incidence_rate: 0.02,
            removal_rate: 0.03,
            recovery_rate: 0.04,
            s_popf: DVector::default(),
            i_popf: DVector::default(),
            r_popf: DVector::default(),
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
            DVector::default(),
            "Bad , expected DVector::default() got {}",
            model.s_popf,
        );
        assert_eq!(
            model.i_popf,
            DVector::default(),
            "Bad , expected DVector::default() got {}",
            model.i_popf,
        );
        assert_eq!(
            model.r_popf,
            DVector::default(),
            "Bad , expected DVector::default() got {}",
            model.r_popf,
        );
    }

    #[test]
    fn test_init_popf() {
        let mut model: Model = Model {
            length: 10,
            i_popf_init: 0.01,
            r_popf_init: 0.0,
            incidence_rate: 0.02,
            removal_rate: 0.03,
            recovery_rate: 0.04,
            s_popf: DVector::default(),
            i_popf: DVector::default(),
            r_popf: DVector::default(),
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
            model.s_popf[0],
            1.0 - model.i_popf_init - model.r_popf_init,
            "Bad s_popf[0] initialization value, expected {} got {}.",
            1.0 - model.i_popf_init - model.r_popf_init,
            model.s_popf[0]
        );
        assert_eq!(
            model.i_popf[0], model.i_popf_init,
            "Bad i_popf[0] initialization value, expected {} got {}.",
            model.i_popf_init, model.i_popf[0],
        );
        assert_eq!(
            model.r_popf[0], model.r_popf_init,
            "Bad r_popf[0] initialization value, expected {} got {}.",
            model.r_popf_init, model.r_popf[0],
        );
        for t in 1..model.length {
            assert_eq!(
                model.s_popf[t], 0.0,
                "Bad s_popf[t>0] initialization value, expected 0.0 got {}.",
                model.s_popf[t]
            );
            assert_eq!(
                model.i_popf[t], 0.0,
                "Bad i_popf[t>0] initialization value, expected 0.0 got {}.",
                model.i_popf[t]
            );
            assert_eq!(
                model.r_popf[t], 0.0,
                "Bad r_popf[t>0] initialization value, expected 0.0 got {}.",
                model.r_popf[t]
            );
        }
    }

    #[test]
    fn test_run_fdm_o1() {
        let mut model: Model = Model {
            length: 10,
            i_popf_init: 0.01,
            r_popf_init: 0.0,
            incidence_rate: 0.02,
            removal_rate: 0.03,
            recovery_rate: 0.04,
            s_popf: DVector::default(),
            i_popf: DVector::default(),
            r_popf: DVector::default(),
        };
        model.init_popf();
        model.run_fdm_o1();
        for t in 1..model.length {
            let dsdt = (-model.incidence_rate * model.s_popf[t - 1] * model.i_popf[t - 1])
                + (model.recovery_rate * model.i_popf[t - 1]);
            let didt = (model.incidence_rate * model.s_popf[t - 1] * model.i_popf[t - 1])
                - (model.removal_rate * model.i_popf[t - 1])
                - (model.recovery_rate * model.i_popf[t - 1]);
            let drdt = model.removal_rate * model.i_popf[t - 1];
            assert!(
                (model.s_popf[t] >= 0.0) & (model.s_popf[t] <= 1.0),
                "s_popf[t] not in [0, 1] at time {}, got {}",
                t, model.s_popf[t]
            );
            assert!(
                (model.i_popf[t] >= 0.0) & (model.i_popf[t] <= 1.0),
                "i_popf[t] not in [0, 1] at time {}, got {}",
                t, model.i_popf[t]
            );
            assert!(
                (model.r_popf[t] >= 0.0) & (model.r_popf[t] <= 1.0),
                "r_popf[t] not in [0, 1] at time {}, got {}",
                t, model.r_popf[t]
            );
            assert_eq!(
                model.s_popf[t], model.s_popf[t - 1] + dsdt,
                "Bad s_popf[t] at time {}, expected {} got {}",
                t, model.s_popf[t - 1] + dsdt, model.s_popf[t]
            );
            assert_eq!(
                model.i_popf[t], model.i_popf[t - 1] + didt,
                "Bad i_popf[t] at time {}, expected {} got {}",
                t, model.i_popf[t - 1] + didt, model.i_popf[t]
            );
            assert_eq!(
                model.r_popf[t], model.r_popf[t - 1] + drdt,
                "Bad r_popf[t] at time {}, expected {} got {}",
                t, model.r_popf[t - 1] + drdt, model.r_popf[t]
            );
        }
    }
}
