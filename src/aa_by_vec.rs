//! all_actions_vec

use crate::{Action, Language, Word};


impl<const A: u8> Word<A> {
	pub fn all_actions_vec(self) -> Vec<Action> {
		use Action::*;

		let len = self.len();
		let alphabet = Language::get_alphabet_from_lang_index(A);

		let mut actions_vec = vec![];

		#[cfg(feature="remove")] // COMPLEXITY: L
		for index in 0..len {
			actions_vec.push(Remove { index });
		}

		#[cfg(feature="take")] // COMPLEXITY: ~ L^2
		for index_start in 0..len {
			for index_end in index_start+1..len {
				actions_vec.push(Take { index_start, index_end });
			}
		}

		#[cfg(feature="discard")] // COMPLEXITY: ~ L^2
		for index_start in 0..len {
			for index_end in index_start+1..len {
				actions_vec.push(Discard { index_start, index_end });
			}
		}

		#[cfg(feature="replace")] // COMPLEXITY: L * A
		for index in 0..len {
			for char in alphabet.chars() {
				if self.chars[index] == char { continue }
				actions_vec.push(Replace { char, index });
			}
		}

		#[cfg(feature="swap")] // COMPLEXITY: ~ L^4
		for index1s in 0..len {
			for index1e in index1s..len {
				for index2s in index1e+1..len {
					for index2e in index2s..len {
						actions_vec.push(Swap { index1s, index1e, index2s, index2e });
					}
				}
			}
		}

		#[cfg(feature="add")] // COMPLEXITY: (L+1) * A
		for index in 0..=len {
			for char in alphabet.chars() {
				actions_vec.push(Add { index, char });
			}
		}

		#[cfg(feature="copy")] // COMPLEXITY: ~ L^3
		for index_start in 0..len {
			for index_end in index_start+2..len {
				for index_insert in 0..=len {
					actions_vec.push(Copy_ { index_start, index_end, index_insert });
				}
			}
		}

		actions_vec.shrink_to_fit();
		actions_vec
	}
}

