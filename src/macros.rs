//! Macros.

#[macro_export]
macro_rules! swap {
    ($a:expr, $b:expr) => { {
        let t = $a;
        $a = $b;
        $b = t;
    } };
}

#[macro_export]
macro_rules! unmut {
    ($x:tt) => {
        let $x = $x;
    };
}

