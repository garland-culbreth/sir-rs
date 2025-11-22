use sirrs::dismod::Model;

#[test]
fn dismod_init_popf() {
    let mut model = Model::new();
    model.configure(
        10,
        1.0,
        0.01,
        0.01,
        0.02,
        0.03,
        0.04,
    );
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
        model.c[(0, 0)], model.c_init,
        "Bad c[(0, 0)] initialization value, expected {} got {}.",
        model.c_init, model.c[(0, 0)],
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
    }
}

#[test]
fn dismod_run_euler() {
    let mut model = Model::new();
    model.configure(
        10,
        1.0,
        0.01,
        0.01,
        0.02,
        0.03,
        0.04,
    );
    model.init_popf();
    model.run_euler();
    for t in 1..model.length {
        let dsdt =
            -((model.iota + model.omega) * model.s[(t - 1, 0)]) + (model.rho * model.c[(t - 1, 0)]);
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
