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

// src: https://internals.rust-lang.org/t/mutually-exclusive-feature-flags/8601/7
#[macro_export]
macro_rules! assert_unique_feature {
	() => {};
	($first:tt $(,$rest:tt)* $(,)?) => {
		$(
			#[cfg(all(feature = $first, feature = $rest))]
			compile_error!(concat!("features `", $first, "` and `", $rest, "` are mutually exlusive"));
		)*
		assert_unique_feature!($($rest),*);
	}
}

