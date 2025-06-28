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
    /// Initial infectious population fraction.
    pub c_init: f64,
    /// Transition rate from S into I. Must be in [0, 1].
    pub iota: f64,
    /// Transition rate from I into S. Must be in [0, 1].
    pub rho: f64,
    /// Transition rate from I into Rc. Must be in [0, 1].
    pub chi: f64,
    /// Transition rate from S, I into Ro. Must be in [0, 1].
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
            let dsdt = (-self.iota * self.s[t - 1])
                + (self.rho * self.c[t - 1]);
            let dcdt = (self.s[t - 1] * self.iota)
                - (self.rho + self.chi + self.omega)
                    * self.c[t - 1];
            let drcdt = self.chi * self.c[t - 1];
            let drodt = (self.omega * self.c[t - 1])
                + (self.omega * self.s[t - 1]);
            self.s[t] = self.s[t - 1] + dsdt;
            self.c[t] = self.c[t - 1] + dcdt;
            self.rc[t] = self.rc[t - 1] + drcdt;
            self.ro[t] = self.ro[t - 1] + drodt;
            println!(
                "t={}: s={:.6} c={:.6} rc={:.6} ro={:.6}",
                t,
                self.s[t],
                self.c[t],
                self.rc[t],
                self.ro[t]
            );
        }
        return self;
    }
}
