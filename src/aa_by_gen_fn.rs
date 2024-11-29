//! all_actions_iter_by_gen_fn

use crate::{Action, Language, Word};


impl<const A: u8> Word<A> {
	pub gen fn all_actions_iter_by_gen_fn(&self) -> Action {
		use Action::*;
		let len = self.len();
		let alphabet = Language::get_alphabet_from_lang_index(A);

		#[cfg(feature="remove")] // COMPLEXITY: L
		for index in 0..len {
			yield Remove { index }
		}

		#[cfg(feature="take")] // COMPLEXITY: ~ L^2
		for index_start in 0..len {
			for index_end in index_start+1..len {
				yield Take { index_start, index_end }
			}
		}

		#[cfg(feature="drop")] // COMPLEXITY: ~ L^2
		for index_start in 0..len {
			for index_end in index_start+1..len {
				yield Drop_ { index_start, index_end }
			}
		}

		#[cfg(feature="replace")] // COMPLEXITY: L * A
		for index in 0..len {
			for char in alphabet.chars() {
				if self.chars[index] == char { continue }
				yield Replace { char, index }
			}
		}

		#[cfg(feature="swap")] // COMPLEXITY: ~ L^4
		for index1s in 0..len {
			for index1e in index1s..len {
				for index2s in index1e+1..len {
					for index2e in index2s..len {
						yield Swap { index1s, index1e, index2s, index2e }
					}
				}
			}
		}

		#[cfg(feature="add")] // COMPLEXITY: (L+1) * A
		for index in 0..=len {
			for char in alphabet.chars() {
				yield Add { index, char }
			}
		}

		#[cfg(feature="copy")] // COMPLEXITY: ~ L^3
		for index_start in 0..len {
			for index_end in index_start+2..len {
				for index_insert in 0..=len {
					yield Copy_ { index_start, index_end, index_insert }
				}
			}
		}
	}
}

