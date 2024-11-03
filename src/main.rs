//! Calc Levenshtein distance between words

#![feature(
	coroutines,
	coroutine_trait,
	iter_from_coroutine,
	stmt_expr_attributes,
)]

use std::{iter::from_coroutine, ops::Add};

use clap::{Parser, arg};

mod extensions;
mod macros;
mod utils_io;

use extensions::VecPushed;
use utils_io::prompt;



#[derive(Parser, Debug)]
#[clap(
	about,
	author,
	version,
	help_template = "\
		{before-help}{name} {version}\n\
		{about}\n\
		Author: {author}\n\
		\n\
		{usage-heading} {usage}\n\
		\n\
		{all-args}{after-help}\
	",
)]
struct CliArgsPre {
	/// Language: `eng` or `ukr`.
	#[arg(short, long, default_value="eng")]
	language_str: String,
}

struct CliArgsPost {
	language: Language,
}
impl From<CliArgsPre> for CliArgsPost {
	fn from(CliArgsPre {
		language_str,
	}: CliArgsPre) -> Self {
		Self {
			language: match language_str.as_str() {
				"eng" => Language::Eng,
				"ukr" => Language::Ukr,
				_ => panic!()
			},
		}
	}
}

enum Language { Eng, Ukr }



fn main() {
	let cli_args = CliArgsPre::parse();
	let cli_args = CliArgsPost::from(cli_args);

	match cli_args.language {
		Language::Eng => {
			let word1 = WordEng::new(&prompt("Input word 1: "));
			let word2 = WordEng::new(&prompt("Input word 2: "));
			let solution = find_solution_st(word1, word2.clone());
			println!("Solution: {solution:#?}");
			let solution_len = solution.len();
			println!("Solution length: {}", solution_len);
			let word2_len = word2.len();
			println!("Length of word 2: {}", word2.chars.len());
			let score_f = calc_score_f(word2_len, solution_len);
			println!("Points float: {score_f}");
			let score = calc_score(word2_len, solution_len);
			println!("Points: {score}");
		}
		Language::Ukr => {
			let word1 = WordUkr::new(&prompt("Введіть слово 1: "));
			let word2 = WordUkr::new(&prompt("Введіть слово 2: "));
			let solution = find_solution_st(word1, word2.clone());
			println!("Розв'язок: {solution:#?}");
			let solution_len = solution.len();
			println!("Довжина розв'язку: {}", solution_len);
			let word2_len = word2.len();
			println!("Довжина слова 2: {}", word2_len);
			let score_f = calc_score_f(word2_len, solution_len);
			println!("Очків float: {score_f}");
			let score = calc_score(word2_len, solution_len);
			println!("Очків: {score}");
		}
	}
}



#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
	/* all features/rules:
	#[cfg(feature = "add")]
	#[cfg(feature = "remove")]
	#[cfg(feature = "replace")]
	#[cfg(feature = "swap_ranges")]
	*/

	#[cfg(feature = "add")]
	Add { char: char, index: usize },

	#[cfg(feature = "remove")]
	Remove { index: usize },

	#[cfg(feature = "replace")]
	Replace { index: usize, char: char },

	// SwapAtIndices { index1: usize, index2: usize },

	#[cfg(feature = "swap_ranges")]
	SwapRanges { index1s: usize, index1e: usize, index2s: usize, index2e: usize },

	// DiscardHeadOrTail { is_head: bool, index: usize },
}

impl Action {
	fn shift_indices_mut(&mut self, shift: usize) {
		use Action::*;
		match self {
			#[cfg(feature = "add")]
			Add { char: _, index } => {
				*index += shift
			}
			#[cfg(feature = "remove")]
			Remove { index } => {
				*index += shift
			}
			#[cfg(feature = "replace")]
			Replace { index, char: _ } => {
				*index += shift
			}
			#[cfg(feature = "swap_ranges")]
			SwapRanges { index1s, index1e, index2s, index2e } => {
				*index1s += shift;
				*index1e += shift;
				*index2s += shift;
				*index2e += shift;
			}
		}
	}

	fn shifted_indices(mut self, shift: usize) -> Self {
		self.shift_indices_mut(shift);
		self
	}
}


const A_ENG_I: u8 = 0;
const A_UKR_I: u8 = 1;
const fn get_alphabet_by_index(index: u8) -> &'static str {
	const ALPHABET_ENG: &str = "abcdefghijklmnopqrstuvwxyz";
	const ALPHABET_UKR: &str = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя";
	match index {
		A_ENG_I => ALPHABET_ENG,
		A_UKR_I => ALPHABET_UKR,
		_ => panic!()
	}
}

type WordEng = Word<A_ENG_I>;
type WordUkr = Word<A_UKR_I>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Word<const A: u8> {
	chars: Vec<char>,
}
impl<const A: u8> Word<A> {
	// const MAX_LEN: usize = 9;

	fn from(chars: Vec<char>) -> Self {
		// if chars.len() == 0 || chars.len() > Self::MAX_LEN { panic!() }
		// if chars.len() == 0 { panic!() }
		assert!(chars.clone().into_iter().all(|c| get_alphabet_by_index(A).contains(c)));
		Self { chars }
	}

	fn new(word_str: &str) -> Self {
		Self::from(word_str.chars().collect())
	}

	#[expect(unused)]
	fn to_string(&self) -> String {
		self.chars.iter().collect()
	}

	fn len(&self) -> usize {
		self.chars.len()
	}

	fn is_legal_action(&self, action: Action) -> bool {
		let self_len = self.len();
		match action {
			#[cfg(feature = "add")]
			Action::Add { index, char: _ } => {
				// if self_len == Self::MAX_LEN { return false }
				if index > self_len { return false }
			}
			#[cfg(feature = "remove")]
			Action::Remove { index } => {
				// if self_len == 1 { return false }
				if index >= self_len { return false }
			}
			#[cfg(feature = "replace")]
			Action::Replace { index, char } => {
				if index > self_len { return false }
				if self.chars[index] == char { return false }
			}
			#[cfg(feature = "swap_ranges")]
			Action::SwapRanges { index1s, index1e, index2s, index2e } => {
				if index1s > self_len { return false }
				if index1e > self_len { return false }
				if index2s > self_len { return false }
				if index2e > self_len { return false }
				if !(index1s <= index1e && index1e < index2s && index2s <= index2e) { return false }
			}
		}
		true
	}

	fn apply_action(&self, action: Action) -> Self {
		let mut self_ = self.clone();
		self_.apply_action_mut(action);
		self_
	}

	fn apply_action_mut(&mut self, action: Action) {
		if !self.is_legal_action(action) { panic!("self={self:?}\naction={action:?}") }
		// dbg!(action);
		match action {
			#[cfg(feature = "add")]
			Action::Add { char, index } => {
				self.chars.insert(index, char);
			}
			#[cfg(feature = "remove")]
			Action::Remove { index } => {
				self.chars.remove(index);
			}
			#[cfg(feature = "replace")]
			Action::Replace { index, char } => {
				self.chars[index] = char;
			}
			#[cfg(feature = "swap_ranges")]
			Action::SwapRanges { index1s, index1e, index2s, index2e } => {
				let before = &self.chars[..index1s];
				let part_1 = &self.chars[index1s..=index1e];
				let middle = &self.chars[index1e+1..index2s];
				let part_2 = &self.chars[index2s..=index2e];
				let after  = &self.chars[index2e+1..];
				self.chars = [before, part_2, middle, part_1, after]
					.iter()
					.map(|p| p.iter().map(|&c| c))
					.flatten()
					.collect();
			}
		}
	}

	fn all_actions(self) -> impl Iterator<Item=Action> {
		from_coroutine(#[coroutine] move || {
			let len = self.len();
			let alphabet = get_alphabet_by_index(A);

			#[cfg(feature = "add")]
			for index in 0..=len {
				for char in alphabet.chars() {
					yield Action::Add { char, index }
				}
			}

			#[cfg(feature = "remove")]
			for index in 0..len {
				yield Action::Remove { index }
			}

			#[cfg(feature = "replace")]
			for index in 0..len {
				for char in alphabet.chars() {
					if self.chars[index] == char { continue }
					yield Action::Replace { char, index }
				}
			}

			#[cfg(feature = "swap_ranges")]
			for index1s in 0..len {
				for index1e in index1s..len {
					for index2s in index1e+1..len {
						for index2e in index2s..len {
							yield Action::SwapRanges { index1s, index1e, index2s, index2e }
						}
					}
				}
			}
		})
	}

	fn dropped_at_index(&self, index: usize) -> Self {
		let mut self_ = self.clone();
		self_.chars.remove(index);
		self_
	}

	fn dropped_first(&self) -> Self {
		self.dropped_at_index(0)
	}

	fn dropped_last(&self) -> Self {
		self.dropped_at_index(self.len()-1)
	}
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrefixSuffixLen { prefix_len: usize, suffix_len: usize }
impl From<(usize, usize)> for PrefixSuffixLen {
	fn from((prefix_len, suffix_len): (usize, usize)) -> Self {
		Self { prefix_len, suffix_len }
	}
}
impl Add<PrefixSuffixLen> for (usize, usize) {
	type Output = PrefixSuffixLen;
	fn add(self, rhs: PrefixSuffixLen) -> Self::Output {
		PrefixSuffixLen {
			prefix_len: self.0 + rhs.prefix_len,
			suffix_len: self.1 + rhs.suffix_len,
		}
	}
}
/// Returns number or commond letter in the begin and end of the word.
fn calc_common_prefix_and_suffix_len<const A: u8>(word1: &Word<A>, word2: &Word<A>) -> PrefixSuffixLen {
	if word1.len() == 0 || word2.len() == 0 { return (0, 0).into() }
	else if word1.chars.first() == word2.chars.first() { // if prefix
		return (1, 0) + calc_common_prefix_and_suffix_len(
			&word1.dropped_first(),
			&word2.dropped_first(),
		)
	}
	else if word1.chars.last() == word2.chars.last() { // if suffix
		return (0, 1) + calc_common_prefix_and_suffix_len(
			&word1.dropped_last(),
			&word2.dropped_last(),
		)
	}
	else {
		return (0, 0).into()
	}
}


fn find_solutions_st<const A: u8>(word_initial: Word<A>, word_target: Word<A>) -> Vec<Vec<Action>> {
	// dbg!(&word_initial, &word_target);
	if word_initial == word_target { return vec![vec![]] }
	let mut words: Vec<(Word<A>, Vec<Action>)> = vec![(word_initial, vec![])];
	let mut new_words: Vec<(Word<A>, Vec<Action>)> = vec![];
	let mut solutions: Vec<Vec<Action>> = vec![];
	let mut search_depth: u8 = 0;
	while solutions.is_empty() {
		search_depth += 1;
		// dbg!(search_depth);
		for (word, actions) in words.into_iter() {
			// dbg!(&word, &actions);
			match calc_common_prefix_and_suffix_len(&word, &word_target) {
				PrefixSuffixLen { prefix_len: 0, suffix_len: 0 } => {
					for action in word.clone().all_actions() {
						let new_word = word.apply_action(action);
						// dbg!(&new_word);
						let new_actions = actions.clone().pushed_opt(action);
						if new_word == word_target { solutions.push(new_actions.clone()) }
						new_words.push((new_word, new_actions));
					}
				}
				PrefixSuffixLen { prefix_len, suffix_len } => {
					// ncp = non common part
					// dbg!(prefix_len, suffix_len);
					let word_ncp = Word::<A>::from(word.chars[prefix_len..word.len()-suffix_len].to_vec());
					let word_target_ncp = Word::from(word_target.chars[prefix_len..word_target.len()-suffix_len].to_vec());
					for solution in find_solutions_st(word_ncp, word_target_ncp) {
						// solutions.push(
						// 	actions
						// 		.clone()
						// 		.into_iter()
						// 		.chain(
						// 			solution
						// 				.iter()
						// 				.map(|a| a.shifted_indices(prefix_len))
						// 		)
						// 		.collect()
						// );

						let mut solution: Vec<Action> = solution
							.iter()
							.map(|a| a.shifted_indices(prefix_len))
							.collect();
						solution.shrink_to_fit();
						let mut actions = actions.clone();
						actions.extend(solution);
						actions.shrink_to_fit();
						solutions.push(actions);
					}
				}
			}
		}
		words = new_words;
		words.shrink_to_fit();
		new_words = vec![];
	}
	solutions
}

fn find_solution_st<const A: u8>(word_initial: Word<A>, word_target: Word<A>) -> Vec<Action> {
	find_solutions_st(word_initial, word_target)
		.into_iter()
		.min_by_key(|s| s.len())
		.unwrap()
}

fn find_solution_mt() {
	todo!()
}



fn calc_score_f(word2_len: usize, solution_len: usize) -> f64 {
	(word2_len as f64) / (solution_len as f64)
}

fn calc_score(word2_len: usize, solution_len: usize) -> u8 {
	calc_score_f(word2_len, solution_len).round() as u8
}



#[cfg(test)]
mod tests {
	use super::*;

	mod find_solution {
		use super::*;

		#[test]
		fn trivial() {
			assert_eq!(
				Vec::<Action>::new(),
				find_solution_st(WordEng::new("foobar"), WordEng::new("foobar"))
			)
		}

		#[cfg(feature = "add")]
		mod add {
			use super::*;
			use Action::Add;
			#[test]
			fn b() {
				assert_eq!(
					vec![
						Add { char: 'x', index: 0 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("xfoobar"))
				)
			}
			#[test]
			fn bb() {
				assert_eq!(
					vec![
						Add { char: 'x', index: 0 },
						Add { char: 'y', index: 0 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("yxfoobar"))
				)
			}
			#[test]
			fn bbb() {
				assert_eq!(
					vec![
						Add { char: 'x', index: 0 },
						Add { char: 'y', index: 0 },
						Add { char: 'z', index: 0 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("zyxfoobar"))
				)
			}
			#[test]
			fn e() {
				assert_eq!(
					vec![
						Add { char: 'x', index: 6 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobarx"))
				)
			}
			#[test]
			fn ee() {
				assert_eq!(
					vec![
						Add { char: 'x', index: 6 },
						Add { char: 'y', index: 7 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobarxy"))
				)
			}
			#[test]
			fn eee() {
				assert_eq!(
					vec![
						Add { char: 'x', index: 6 },
						Add { char: 'y', index: 7 },
						Add { char: 'z', index: 8 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobarxyz"))
				)
			}
			#[test]
			fn m() {
				assert_eq!(
					vec![
						Add { char: 'x', index: 3 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fooxbar"))
				)
			}
			#[test]
			fn mm() {
				assert_eq!(
					vec![
						Add { char: 'x', index: 3 },
						Add { char: 'y', index: 4 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fooxybar"))
				)
			}
			#[test]
			fn mmm() {
				assert_eq!(
					vec![
						Add { char: 'x', index: 3 },
						Add { char: 'y', index: 4 },
						Add { char: 'z', index: 5 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fooxyzbar"))
				)
			}
		}

		#[cfg(feature = "remove")]
		mod remove {
			use super::*;
			use Action::Remove;
			#[test]
			fn b() {
				assert_eq!(
					vec![
						Remove { index: 0 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("oobar"))
				)
			}
			#[test]
			fn bb() {
				assert_eq!(
					vec![
						Remove { index: 0 },
						Remove { index: 0 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("obar"))
				)
			}
			#[test]
			fn bbb() {
				assert_eq!(
					vec![
						Remove { index: 0 },
						Remove { index: 0 },
						Remove { index: 0 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("bar"))
				)
			}
			#[test]
			fn e() {
				assert_eq!(
					vec![
						Remove { index: 5 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fooba"))
				)
			}
			#[test]
			fn ee() {
				assert!(
					vec![ // solutions:
						vec![
							Remove { index: 5 },
							Remove { index: 4 },
						],
						vec![
							Remove { index: 4 },
							Remove { index: 4 },
						],
					].contains(
						&find_solution_st(WordEng::new("foobar"), WordEng::new("foob"))
					)
				)
			}
			#[test]
			fn eee() {
				assert!(
					vec![ // solutions:
						vec![
							Remove { index: 5 },
							Remove { index: 4 },
							Remove { index: 3 },
						],
						vec![
							Remove { index: 3 },
							Remove { index: 3 },
							Remove { index: 3 },
						],
					].contains(
						&find_solution_st(WordEng::new("foobar"), WordEng::new("foo"))
					)
				)
			}
		}

		#[cfg(feature = "replace")]
		mod replace {
			use super::*;
			use Action::Replace;
			#[test]
			fn b() {
				assert_eq!(
					vec![
						Replace { index: 0, char: 'x' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("xoobar"))
				)
			}
			#[test]
			fn e() {
				assert_eq!(
					vec![
						Replace { index: 5, char: 'x' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobax"))
				)
			}
			#[test]
			fn m() {
				assert_eq!(
					vec![
						Replace { index: 2, char: 'x' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foxbar"))
				)
			}
		}

		#[cfg(feature = "swap_ranges")]
		mod swap_ranges {
			use super::*;
			use Action::SwapRanges;
			#[test]
			fn foobar_barfoo() {
				assert_eq!(
					vec![
						SwapRanges { index1s: 0, index1e: 2, index2s: 3, index2e: 5 },
					], //                          012345                  012345
					find_solution_st(WordEng::new("foobar"), WordEng::new("barfoo"))
				)
			}
			#[test]
			fn abcfoodefbarxyz_abcbardeffooxyz() {
				assert_eq!(
					vec![
						SwapRanges { index1s: 3, index1e: 5, index2s: 9, index2e: 11 },
					], //                          012345678901234                  012345678901234
					find_solution_st(WordEng::new("abcfoodefbarxyz"), WordEng::new("abcbardeffooxyz"))
				)
			}
		}
	}

	mod calc_common_prefix_and_suffix_len {
		use super::*;
		#[test]
		fn abcxyzdefgh_abcvdefgh() {
			assert_eq!(
				PrefixSuffixLen { prefix_len: 3, suffix_len: 5 },
				calc_common_prefix_and_suffix_len(
					&WordEng::new("abcxyzdefgh"),
					&WordEng::new("abcvdefgh")
				)
			)
		}
		#[test]
		fn kzko_ko() {
			let expected_solutions = [
				PrefixSuffixLen { prefix_len: 0, suffix_len: 2 },
				PrefixSuffixLen { prefix_len: 1, suffix_len: 1 },
			];
			let actual_solution = calc_common_prefix_and_suffix_len(
				&WordEng::new("kzko"),
				&WordEng::new("ko")
			);
			dbg!(expected_solutions);
			dbg!(actual_solution);
			assert!(expected_solutions.contains(&actual_solution))
		}
		#[test]
		fn kzz_k() {
			assert_eq!(
				PrefixSuffixLen { prefix_len: 1, suffix_len: 0 },
				calc_common_prefix_and_suffix_len(
					&WordEng::new("kzz"),
					&WordEng::new("k")
				)
			)
		}
		#[test]
		fn zzk_k() {
			assert_eq!(
				PrefixSuffixLen { prefix_len: 0, suffix_len: 1 },
				calc_common_prefix_and_suffix_len(
					&WordEng::new("zzk"),
					&WordEng::new("k")
				)
			)
		}
	}
}

