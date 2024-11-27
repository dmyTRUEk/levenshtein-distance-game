//! all_actions_vec_sorted_by_priority

use crate::{Action, Word};


impl<const A: u8> Word<A> {
	pub fn all_actions_vec_sorted_by_priority(self) -> Vec<Action> {
		let mut actions_vec = self.all_actions_vec();
		sort_by_priority(&mut actions_vec);
		actions_vec.shrink_to_fit();
		actions_vec
	}
}

fn sort_by_priority(v: &mut Vec<Action>) {
	v.sort_by_key(|a| a.priority());
}

fn sorted_by_priority(mut v: Vec<Action>) -> Vec<Action> {
	sort_by_priority(&mut v);
	v
}

impl Action {
	fn priority(self) -> i32 {
		use Action::*;
		match self {
			#[cfg(feature="add")]
			Add { .. } => 1,

			#[cfg(feature="remove")]
			Remove { .. } => -1,

			#[cfg(feature="replace")]
			Replace { .. } => 0,

			#[cfg(feature="swap")]
			Swap { index1s, index1e, index2s, index2e } => ((index1e as i32)-(index1s as i32)+1) * ((index2e as i32)-(index2s as i32)+1),

			#[cfg(feature="discard")]
			Discard { index_start, index_end } => -((index_end as i32)-(index_start as i32)+1),

			#[cfg(feature="take")]
			Take { index_start, index_end } => -((index_end as i32)-(index_start as i32)+1),

			#[cfg(feature="copy")]
			Copy_ { index_start, index_end, index_insert } => todo!(),
		}
	}

}





#[cfg(test)]
mod action {
	use super::*;
	mod sort_by_priority {
		use super::*;
		use Action::*;
		#[cfg(all(feature="add", feature="remove", feature="replace"))]
		#[test]
		fn add_remove_replace() {
			assert_eq!(
				vec![
					Remove { index: 0 },
					Replace { index: 0, char: 'x' },
					Action::Add { index: 0, char: 'x' },
				],
				sorted_by_priority(vec![
					Action::Add { index: 0, char: 'x' },
					Remove { index: 0 },
					Replace { index: 0, char: 'x' },
				])
			)
		}
		#[cfg(all(feature="add", feature="remove", feature="replace", feature="swap"))]
		#[test]
		fn add_remove_replace_swap() {
			assert_eq!(
				vec![
					Remove { index: 0 },
					Replace { index: 0, char: 'x' },
					Action::Add { index: 0, char: 'x' },
					Swap { index1s: 0, index1e: 2, index2s: 3, index2e: 5 },
				],
				sorted_by_priority(vec![
					Action::Add { index: 0, char: 'x' },
					Remove { index: 0 },
					Swap { index1s: 0, index1e: 2, index2s: 3, index2e: 5 },
					Replace { index: 0, char: 'x' },
				])
			)
		}
	}
}

