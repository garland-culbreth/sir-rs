//! Three compartment SIR model and methods.
//! 
//! Allows transition rates:
//!  - S → I  
//!  - I → R  
//!  - R → S  
use ndarray::Array;
use ndarray::prelude::*;

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
    pub s_popf: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>,
    /// Inectious population fraction at each index. 1D Array with `length` number of elements.
    pub i_popf: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>,
    /// Removed population fraction at each index. 1D Array with `length` number of elements.
    pub r_popf: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>,
}

impl Model {
    /// Initialize population fractions. Creates arrays of length `self.length`
    /// to store the population fractions at each index and sets the 0th index
    /// of each equal to the corresponding initial population fraction.
    pub fn init_popf(&mut self) -> &mut Model {
        self.s_popf = Array::<f64, _>::zeros((self.length).f());
        self.i_popf = Array::<f64, _>::zeros((self.length).f());
        self.r_popf = Array::<f64, _>::zeros((self.length).f());
        let s_init = 1.0 - self.i_popf_init - self.r_popf_init; // Population fractions must sum to 1.
        self.s_popf[[0]] = s_init;
        self.i_popf[[0]] = self.i_popf_init;
        self.r_popf[[0]] = self.r_popf_init;
        return self;
    }

    /// Run the SIR differential equations by the first-order finite difference
    /// method.
    ///
    /// This solution method is very rough and only suitable for demonstration.
    pub fn run_fdm_o1(&mut self) -> &Model {
        for t in 1..self.s_popf.len() {
            let dsdt = (-self.incidence_rate * self.s_popf[[t - 1]] * self.i_popf[[t - 1]])
                + (self.recovery_rate * self.i_popf[[t - 1]]);
            let didt = (self.incidence_rate * self.s_popf[[t - 1]] * self.i_popf[[t - 1]])
                - (self.removal_rate * self.i_popf[[t - 1]])
                - (self.recovery_rate * self.i_popf[[t - 1]]);
            let drdt = self.removal_rate * self.i_popf[[t - 1]];
            self.s_popf[[t]] = self.s_popf[[t - 1]] + dsdt;
            self.i_popf[[t]] = self.i_popf[[t - 1]] + didt;
            self.r_popf[[t]] = self.r_popf[[t - 1]] + drdt;
            println!(
                "t={}: s={:.6} i={:.6} r={:.6}",
                t,
                self.s_popf[[t]],
                self.i_popf[[t]],
                self.r_popf[[t]]
            );
        }
        return self;
    }
}
