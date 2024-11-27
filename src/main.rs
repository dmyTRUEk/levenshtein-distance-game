//! Calc Levenshtein distance between words.

#![deny(
	unsafe_code,
	unused_results,
)]

// TODO
// #![feature(
// 	let_chains,
// 	variant_count, // TODO
// )]

#![cfg_attr(
	feature="aa_by_coroutine",
	feature(coroutines, coroutine_trait, iter_from_coroutine),
)]

#![cfg_attr(
	any(feature="aa_by_gen_block", feature="aa_by_gen_fn"),
	feature(gen_blocks),
)]



use std::{
	// mem::variant_count as enum_variant_count, // TODO
	ops::Add,
};

use clap::{Parser, arg};
use rand::{rngs::ThreadRng, thread_rng, Rng};

mod extensions;
mod macros;
mod utils_io;

#[cfg(feature="aa_by_coroutine")]
mod aa_by_coroutine;

#[cfg(any(feature="aa_by_vec", feature="aa_by_vec_sbp"))]
mod aa_by_vec;

#[cfg(feature="aa_by_gen_block")]
mod aa_by_gen_block;

#[cfg(feature="aa_by_vec_sbp")]
mod aa_by_vec_sorted_by_priority;

#[cfg(feature="aa_by_gen_fn")]
mod aa_by_gen_fn;

use extensions::{VecPushed, ToStrings};
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
	language: String,

	/// <Word 1>,<Word 2>
	#[arg(short, long)]
	word12: Option<String>,

	/// Search depthes: <random search depth>,<bruteforce search depth>
	#[arg(short='d', long)]
	search_depthes: Option<String>,
}

struct CliArgsPost {
	language: Language,
	word12: Option<[String; 2]>,
	search_depthes: Option<(u8, u8)>,
}
impl From<CliArgsPre> for CliArgsPost {
	fn from(CliArgsPre {
		language,
		word12,
		search_depthes,
	}: CliArgsPre) -> Self {
		Self {
			language: match language.as_str() {
				"eng" => Language::Eng,
				"ukr" => Language::Ukr,
				_ => panic!()
			},
			word12: word12
				.map(|word12| {
					word12.split_once([',', '.', '/', '~']).unwrap().to_strings().into()
				}),
			search_depthes: search_depthes
				.map(|search_depthes| {
					let (sd1, sd2) = search_depthes.split_once([',', '.']).unwrap();
					let sd1: u8 = sd1.parse().unwrap();
					let sd2: u8 = sd2.parse().unwrap();
					(sd1, sd2)
				}),
		}
	}
}



#[cfg(not(any(
	feature="aa_by_coroutine",
	feature="aa_by_vec",
	feature="aa_by_gen_block",
	feature="aa_by_vec_sbp",
	feature="aa_by_gen_fn",
)))]
compile_error!("One of `aa_by_*` features must be enabled");

assert_unique_feature!(
	"aa_by_coroutine",
	"aa_by_vec",
	"aa_by_gen_block",
	"aa_by_vec_sbp",
	"aa_by_gen_fn",
);



fn main() {
	let cli_args = CliArgsPre::parse();
	let cli_args = CliArgsPost::from(cli_args);

	match cli_args.language {
		Language::Eng => {
			main_with_lang::<{Language::ENG}>(cli_args);
		}
		Language::Ukr => {
			main_with_lang::<{Language::UKR}>(cli_args);
		}
	}
}

fn main_with_lang<const A: u8>(cli_args: CliArgsPost) {
	struct Localization {
		word: &'static str,
		input_word: &'static str,
		solution: &'static str,
		solution_len: &'static str,
		word_len: &'static str,
		points_float: &'static str,
		points: &'static str,
	}
	impl Localization {
		const fn get<const A: u8>() -> Self {
			match Language::from_index(A) {
				Language::Eng => Self {
					word: "Word",
					input_word: "Input word",
					solution: "Solution",
					solution_len: "Solution len",
					word_len: "Word len",
					points_float: "Points float",
					points: "Points",
				},
				Language::Ukr => Self {
					word: "Слово",
					input_word: "Введіть слово",
					solution: "Розв'язок",
					solution_len: "Довжина розв'язку",
					word_len: "Довжина слова",
					points_float: "Очків float",
					points: "Очків",
				}
			}
		}
	}

	fn get_word<const A: u8>(
		cli_args: &CliArgsPost,
		text_info_word: &str,
		text_prompt_word: &str,
		n: usize,
	) -> Word<A> {
		Word::new(&match &cli_args.word12 {
			Some(word12) => {
				let word_n_string = &word12[n-1];
				println!("{text_info_word} {n}: {word_n_string}");
				word_n_string.to_string()
			}
			None => prompt(&format!("{text_prompt_word} {n}: "))
		})
	}

	let loc = Localization::get::<A>();
	let get_word_lang = |n: usize| -> Word<A> {
		get_word(&cli_args, loc.word, loc.input_word, n)
	};
	let word1 = get_word_lang(1);
	let word2 = get_word_lang(2);
	let word2_len = word2.len();
	let solution = find_solution_st(word1, word2, cli_args.search_depthes);
	println!("{}: {solution:#?}", loc.solution);
	let solution_len = solution.len();
	println!("{}: {solution_len}", loc.solution_len);
	println!("{} 2: {word2_len}", loc.word_len);
	let score_f = calc_score_f(word2_len, solution_len);
	println!("{}: {score_f}", loc.points_float);
	let score = calc_score(word2_len, solution_len);
	println!("{}: {score}", loc.points);
}



#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Allowed Actions aka Rules
enum Action {
	/* all features/rules:
	#[cfg(feature="add")]
	#[cfg(feature="remove")]
	#[cfg(feature="replace")]
	#[cfg(feature="swap")]
	#[cfg(feature="discard")]
	#[cfg(feature="take")]
	#[cfg(feature="copy")]
	*/

	#[cfg(feature="add")]
	Add { index: usize, char: char },

	#[cfg(feature="remove")]
	Remove { index: usize },

	#[cfg(feature="replace")]
	Replace { index: usize, char: char },

	// #[cfg(feature="swap_one")]
	// SwapAtIndices { index1: usize, index2: usize },

	#[cfg(feature="swap")]
	/// start and end indices are including
	Swap { index1s: usize, index1e: usize, index2s: usize, index2e: usize },

	#[cfg(feature="discard")]
	/// start and end indices are including
	Discard { index_start: usize, index_end: usize },

	#[cfg(feature="take")]
	/// start and end indices are including
	Take { index_start: usize, index_end: usize },

	#[cfg(feature="copy")]
	/// start and end indices are including
	Copy_ { index_start: usize, index_end: usize, index_insert: usize },
}

impl Action {
	fn enum_variant_count() -> usize {
		let mut n: usize = 0;
		#[cfg(feature="add")]     { n += 1 }
		#[cfg(feature="remove")]  { n += 1 }
		#[cfg(feature="replace")] { n += 1 }
		#[cfg(feature="swap")]    { n += 1 }
		#[cfg(feature="discard")] { n += 1 }
		#[cfg(feature="take")]    { n += 1 }
		#[cfg(feature="copy")]    { n += 1 }
		n
	}

	fn shift_indices_mut(&mut self, shift: usize) {
		use Action::*;
		match self {
			#[cfg(feature="add")]
			Add { index, char: _ } => {
				*index += shift;
			}
			#[cfg(feature="remove")]
			Remove { index } => {
				*index += shift;
			}
			#[cfg(feature="replace")]
			Replace { index, char: _ } => {
				*index += shift;
			}
			#[cfg(feature="swap")]
			Swap { index1s, index1e, index2s, index2e } => {
				*index1s += shift;
				*index1e += shift;
				*index2s += shift;
				*index2e += shift;
			}
			#[cfg(feature="discard")]
			Discard { index_start, index_end } => {
				*index_start += shift;
				*index_end   += shift;
			}
			#[cfg(feature="take")]
			Take { index_start, index_end } => {
				*index_start += shift;
				*index_end   += shift;
			}
			#[cfg(feature="copy")]
			Copy_ { index_start, index_end, index_insert } => {
				*index_start  += shift;
				*index_end    += shift;
				*index_insert += shift;
			}
			#[allow(unreachable_patterns)] // reason: to test if works with `--no-default-features`
			_ => {}
		}
	}

	fn shifted_indices(mut self, shift: usize) -> Self {
		self.shift_indices_mut(shift);
		self
	}

	fn is_vain<const A: u8>(&self, word1_len: usize, word2_len: usize) -> bool {
		use Action::*;
		let l1 = word1_len;
		let l2 = word2_len;
		match self {
			// OPTIMIZATIONS 1:

			// // abcx -> zabc:
			// // 1. add{0,x} , remove{4} => 2 ops
			// // 2. remove{3} , add{0,x} => 2 ops
			// // 3. swap{0,2,3,3} => 1 op !!!
			// #[cfg(all(feature="add", feature="remove", feature="swap"))]
			// Add { .. } if l1 == l2 => true,

			#[cfg(feature="add")]
			Add { .. } if l2 == 0 => true,
			#[cfg(feature="remove")]
			Remove { .. } if l2 == 0 => false, // to avoid mistakes
			#[cfg(feature="replace")]
			Replace { .. } if l2 == 0 => true,
			#[cfg(feature="swap")]
			Swap { .. } if l2 == 0 => true,

			#[cfg(feature="discard")]
			Discard { .. } if l1 <= 1 => true,
			#[cfg(feature="take")]
			Take { .. } if l1 <= 2 => true,
			#[cfg(feature="copy")]
			Copy_ { .. } if l1 <= 1 || l2 <= 1 => true,

			#[cfg(feature="copy")]
			Copy_ { .. } if l1 == 2 && l2 < 4 => true, // TODO: recheck

			// TODO: more?

			_ => false
		}
	}

	/// `action_prev.is_stupid_before(action_next)`
	/// returns if `action_next` is stupid after `action_prev`.
	fn is_vain_with(&self, action_next: &Action) -> bool {
		use Action::*;
		match (self, action_next) {
			// OPTIMIZATIONS 2:

			#[cfg(feature="replace")]
			(Replace { index: i1, .. }, Replace { index: i2, .. }) if i1 == i2 => true,

			#[cfg(all(feature="add", feature="replace"))]
			(Add { index: i1, .. }, Replace { index: i2, .. }) if i1 == i2 => true,

			#[cfg(all(feature="add", feature="remove"))]
			(Add { index: i1, .. }, Remove { index: i2 }) if i1 == i2 => true,

			_ => false
		}
	}

	fn is_copy(&self) -> bool {
		match self {
			#[cfg(feature="copy")]
			Self::Copy_ { .. } => { true }
			_ => false
		}
	}
}



enum Language { Eng, Ukr }
impl Language {
	const ENG: u8 = 0;
	const UKR: u8 = 1;
	pub const fn from_index(lang_index: u8) -> Self {
		use Language::*;
		match lang_index {
			Self::ENG => Eng,
			Self::UKR => Ukr,
			_ => panic!()
		}
	}
	pub const fn get_alphabet_from_lang_index(lang_index: u8) -> &'static str {
		Self::from_index(lang_index).get_alphabet()
	}
	pub const fn get_alphabet(&self) -> &'static str {
		use Language::*;
		const ALPHABET_ENG: &str = "abcdefghijklmnopqrstuvwxyz";
		const ALPHABET_UKR: &str = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'";
		match self {
			Eng => ALPHABET_ENG,
			Ukr => ALPHABET_UKR,
		}
	}
	// pub fn get_random_char(lang_index: u8) -> char {
	// 	Self::get_random_char_with_rng(lang_index, &mut thread_rng())
	// }
	pub fn get_random_char_with_rng(lang_index: u8, rng: &mut ThreadRng) -> char {
		let alphabet = Self::get_alphabet_from_lang_index(lang_index);
		alphabet.chars()
			.nth(rng.gen_range(0..alphabet.len()))
			.unwrap()
	}
}




#[derive(Debug, Clone, PartialEq, Eq)]
struct Word<const A: u8> {
	chars: Vec<char>,
}
impl<const A: u8> Word<A> {
	// const MAX_LEN: usize = 9;

	fn from(chars: &[char]) -> Self {
		// if chars.len() == 0 || chars.len() > Self::MAX_LEN { panic!() }
		// if chars.len() == 0 { panic!() }
		let alphabet = Language::get_alphabet_from_lang_index(A);
		assert!(chars.into_iter().all(|&c| alphabet.contains(c)));
		let mut chars = chars.to_vec();
		chars.shrink_to_fit();
		Self { chars }
	}

	fn new(word_str: &str) -> Self {
		Self::from(&word_str.chars().collect::<Vec<char>>())
	}

	#[expect(unused)]
	fn to_string(&self) -> String {
		self.chars.iter().collect()
	}

	fn len(&self) -> usize {
		self.chars.len()
	}

	fn is_legal_action(&self, action: Action) -> bool {
		use Action::*;
		let self_len = self.len();
		match action {
			#[cfg(feature="add")]
			Add { index, char: _ } => {
				// if self_len == Self::MAX_LEN { return false }
				if index > self_len { return false }
			}
			#[cfg(feature="remove")]
			Remove { index } => {
				// if self_len == 1 { return false }
				if index >= self_len { return false }
			}
			#[cfg(feature="replace")]
			Replace { index, char } => {
				if index >= self_len { return false }
				if self.chars[index] == char { return false }
			}
			#[cfg(feature="swap")]
			Swap { index1s, index1e, index2s, index2e } => {
				if index1s >= self_len { return false }
				if index1e >= self_len { return false }
				if index2s >= self_len { return false }
				if index2e >= self_len { return false }
				if !(index1s <= index1e && index1e < index2s && index2s <= index2e) { return false }
			}
			#[cfg(feature="discard")]
			Discard { index_start, index_end } => {
				if index_start >= self_len { return false }
				if index_end   >= self_len { return false }
				if !(index_start <= index_end) { return false }
			}
			#[cfg(feature="take")]
			Take { index_start, index_end } => {
				if index_start >= self_len { return false }
				if index_end   >= self_len { return false }
				if !(index_start <= index_end) { return false }
			}
			#[cfg(feature="copy")]
			Copy_ { index_start, index_end, index_insert } => {
				if index_start  >= self_len { return false }
				if index_end    >= self_len { return false }
				if index_insert >  self_len { return false }
				if !(index_start + 1 < index_end) { return false }
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
		// dbg!(&self, action);
		use Action::*;
		if !self.is_legal_action(action) { panic!("self={self:?}\naction={action:?}") }
		// dbg!(action);
		match action {
			#[cfg(feature="add")]
			Add { index, char } => {
				self.chars.insert(index, char);
			}
			#[cfg(feature="remove")]
			Remove { index } => {
				let _ = self.chars.remove(index);
			}
			#[cfg(feature="replace")]
			Replace { index, char } => {
				self.chars[index] = char;
			}
			#[cfg(feature="swap")]
			Swap { index1s, index1e, index2s, index2e } => {
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
			#[cfg(feature="discard")]
			Discard { index_start, index_end } => {
				let _ = self.chars.drain(index_start..=index_end);
			}
			#[cfg(feature="take")]
			Take { index_start, index_end } => {
				self.chars = self.chars[index_start..=index_end].to_vec();
			}
			#[cfg(feature="copy")]
			Copy_ { index_start, index_end, index_insert } => {
				let _ = self.chars.splice(
					index_insert..index_insert,
					self.chars[index_start..=index_end].to_vec()
				);
			}
		}
	}

	fn dropped_at_index(&self, index: usize) -> Self {
		let mut self_ = self.clone();
		let _ = self_.chars.remove(index);
		self_
	}

	fn dropped_first(&self) -> Self {
		self.dropped_at_index(0)
	}

	fn dropped_last(&self) -> Self {
		self.dropped_at_index(self.len()-1)
	}

	fn gen_random_legal_action(&self) -> Action {
		loop {
			let random_action = self.gen_random_action();
			if self.is_legal_action(random_action) {
				return random_action;
			}
		}
	}

	fn gen_random_action(&self) -> Action {
		use Action::*;
		// println!("{}", "-".repeat(42));
		let mut rng = thread_rng();
		macro_rules! random { ($range:expr) => { rng.gen_range($range) } }
		macro_rules! random_char { () => { Language::get_random_char_with_rng(A, &mut rng) } }
		let len = self.len();
		// dbg!(len);
		macro_rules! random_index    { () => { random!(0..len)   } }
		macro_rules! random_index_p1 { () => { random!(0..=len) } }

		macro_rules! random_add {
			() => {
				Add {
					index: random_index_p1!(),
					char: random_char!(),
				}
			};
		}

		if len == 0 { // see #c5ef13
			#[cfg(feature="add")]
			return random_add!();
			#[allow(unreachable_code)]
			{unreachable!("any other action with len==0 is impossible")}
		}

		macro_rules! random_index_m1 { () => { random!(0..len-1) } }
		macro_rules! random_index_m2 { () => { random!(0..len-2) } }

		// let mut random_action_index = random!(0..enum_variant_count::<Action>()); // TODO
		let mut random_action_index = random!(0..{
			let mut max = Action::enum_variant_count();
			match len {
				0 => unreachable!(), // checked above, see #c5ef13
				1 => {
					#[cfg(feature="swap")] { max -= 1 }
					#[cfg(feature="discard")] { max -= 1 }
					#[cfg(feature="take")] { max -= 1 }
					#[cfg(feature="copy")] { max -= 1 }
				}
				2 => {}
				_ => {}
			}
			max
		});
		// dbg!(random_action_index);


		assert!(random_action_index < Action::enum_variant_count());

		// dbg!(random_action_index);
		// dbg!();
		#[cfg(feature="add")] {
			if random_action_index == 0 {
				return random_add!();
			}
			random_action_index -= 1;
		}
		// dbg!();
		#[cfg(feature="remove")] {
			if random_action_index == 0 {
				return Remove {
					index: random_index!(),
				};
			}
			random_action_index -= 1;
		}
		// dbg!();
		#[cfg(feature="replace")] {
			if random_action_index == 0 {
				return Replace {
					index: random_index!(),
					char: random_char!(),
				};
			}
			random_action_index -= 1;
		}
		// dbg!();
		#[cfg(feature="swap")] {
			assert!(len >= 2);
			if random_action_index == 0 {
				let index1s = random_index_m1!();
				// dbg!(index1s);
				let index1e = random!(index1s..len).min(len-2);
				// dbg!(index1e);
				let index2s = random!(index1e+1..len);
				// dbg!(index2s);
				let index2e = random!(index2s..len);
				// dbg!(index2e);
				return Swap { index1s, index1e, index2s, index2e };
			}
			random_action_index -= 1;
		}
		// dbg!();
		#[cfg(feature="discard")] {
			if random_action_index == 0 {
				let index_start = random_index_m1!();
				// dbg!(index_start);
				let index_end = random!(index_start+1..len);
				// dbg!(index_end);
				return Discard { index_start, index_end };
			}
			random_action_index -= 1;
		}
		// dbg!();
		#[cfg(feature="take")] {
			if random_action_index == 0 {
				let index_start = random_index_m1!();
				// dbg!(index_start);
				let index_end = random!(index_start+1..len);
				// dbg!(index_end);
				return Take { index_start, index_end };
			}
			random_action_index -= 1;
		}
		// dbg!();
		#[cfg(feature="copy")] {
			assert!(len >= 2);
			if random_action_index == 0 {
				let index_start = if len == 2 { 0 } else { random_index_m2!() };
				// dbg!(index_start);
				let index_end = if len == 2 { 2 } else { random!(index_start+2..len) };
				// dbg!(index_end);
				let index_insert = random_index_p1!();
				// dbg!(index_insert);
				return Copy_ { index_start, index_end, index_insert };
			}
			random_action_index -= 1;
		}
		unreachable!()
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


fn find_solutions_st<const A: u8>(
	word_initial: Word<A>,
	word_target: Word<A>,
	mut search_depth_left: Option<u8>,
) -> Vec<Vec<Action>> {
	// println!("{}", "-".repeat(42));
	// println!("[{f}:{l}] initial: {}\t\ttarget: {}", word_initial.to_string(), word_target.to_string(), f=file!(), l=line!());
	// dbg!(search_depth_left);
	if word_initial == word_target { return vec![vec![]] }
	let mut words: Vec<(Word<A>, Vec<Action>)> = vec![(word_initial, vec![])];
	let mut new_words: Vec<(Word<A>, Vec<Action>)> = vec![];
	let mut solutions: Vec<Vec<Action>> = vec![];
	while solutions.is_empty() && search_depth_left.is_none_or(|sdl| sdl > 0) {
		// dbg!(search_depth_left);
		if let Some(ref mut sdl) = search_depth_left {
			// dbg!(&sdl);
			*sdl -= 1;
			// dbg!(&sdl);
		}
		for (word, actions) in words.into_iter() {
			// dbg!(&word, &word_target, &actions);
			macro_rules! all_actions {
				() => {{
					#[cfg(feature="aa_by_coroutine")]
					let iter = word.clone().all_actions_iter_by_coroutine();
					#[cfg(feature="aa_by_vec")]
					let iter = word.clone().all_actions_vec();
					#[cfg(feature="aa_by_gen_block")]
					let iter = word.clone().all_actions_iter_by_gen_block();
					#[cfg(feature="aa_by_vec_sbp")]
					let iter = word.clone().all_actions_vec_sorted_by_priority();
					#[cfg(feature="aa_by_gen_fn")]
					let iter = word.clone().all_actions_iter_by_gen_fn();
					iter.into_iter()
				}};
			}
			macro_rules! apply_action_and_update {
				($action:ident) => {
					let new_word = word.apply_action($action);
					// dbg!(&new_word);
					let new_actions = actions.clone().pushed_opt($action);
					if new_word == word_target { solutions.push(new_actions.clone()) }
					new_words.push((new_word, new_actions));
				};
			}
			#[cfg(feature="copy")] {
				for action in all_actions!().filter(|a| a.is_copy()) {
					// TODO(optim): add `is_vain*` checks?
					apply_action_and_update!(action);
				}
			}
			match calc_common_prefix_and_suffix_len(&word, &word_target) {
				PrefixSuffixLen { prefix_len: 0, suffix_len: 0 } => {
					for action in all_actions!().filter(|a| !a.is_copy()) {
						// use optimizations 1:
						if action.is_vain::<A>(word.len(), word_target.len()) { continue }
						// use optimizations 2:
						// if let Some(action_prev) = actions.last() && action_prev.is_vain_with(&action) { continue } // TODO
						if actions.last().is_some_and(|action_prev| action_prev.is_vain_with(&action)) { continue }
						apply_action_and_update!(action);
					}
				}
				PrefixSuffixLen { prefix_len, suffix_len } => {
					// ncp = non common part
					// dbg!(&word, &word_target);
					// dbg!(prefix_len, suffix_len);
					let word_ncp = Word::<A>::from(&word.chars[prefix_len..word.len()-suffix_len]);
					let word_target_ncp = Word::from(&word_target.chars[prefix_len..word_target.len()-suffix_len]);
					new_words.shrink_to_fit();
					for solution in find_solutions_st(word_ncp, word_target_ncp, search_depth_left.map(|sdl| sdl+1)) {
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

fn find_solution_st<const A: u8>(
	word_initial: Word<A>,
	word_target: Word<A>,
	search_depthes: Option<(u8, u8)>,
) -> Vec<Action> {
	// println!("[{f}:{l}] initial: {}\t\ttarget: {}", word_initial.to_string(), word_target.to_string(), f=file!(), l=line!());
	if let Some((random_search_depth, bruteforce_search_depth)) = search_depthes {
		loop {
			let mut random_actions = Vec::<Action>::with_capacity(random_search_depth as usize);
			let mut word_initial = word_initial.clone();
			for _rsd in 0..random_search_depth {
				let random_action = word_initial.gen_random_legal_action();
				word_initial.apply_action_mut(random_action);
				random_actions.push(random_action);
				if word_initial == word_target {
					return random_actions;
				}
			}
			// dbg!(&word_initial);
			let solution = find_solutions_st(word_initial, word_target.clone(), Some(bruteforce_search_depth))
				.into_iter()
				.min_by_key(|s| s.len());
			if let Some(s) = solution {
				let mut actions = random_actions;
				actions.extend(s);
				return actions;
			}
		}
	}
	else {
		find_solutions_st(word_initial, word_target, None)
			.into_iter()
			.min_by_key(|s| s.len())
			.unwrap()
	}
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

	type WordEng = Word<{Language::ENG}>;
	// type WordUkr = Word<{Language::UKR}>;

	mod find_solution {
		use super::*;

		#[test]
		fn trivial() {
			assert_eq!(
				Vec::<Action>::new(),
				find_solution_st(WordEng::new("foobar"), WordEng::new("foobar"), None)
			)
		}

		#[cfg(feature="add")]
		mod add {
			use super::*;
			use Action::Add;
			#[test]
			fn b() {
				assert_eq!(
					vec![Add { index: 0, char: 'x' }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("xfoobar"), None)
				)
			}
			#[test]
			fn bb() {
				assert_eq!(
					vec![
						Add { index: 0, char: 'x' },
						Add { index: 0, char: 'y' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("yxfoobar"), None)
				)
			}
			#[test]
			fn bbb() {
				assert_eq!(
					vec![
						Add { index: 0, char: 'x' },
						Add { index: 0, char: 'y' },
						Add { index: 0, char: 'z' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("zyxfoobar"), None)
				)
			}
			#[test]
			fn e() {
				assert_eq!(
					vec![Add { index: 6, char: 'x' }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobarx"), None)
				)
			}
			#[test]
			fn ee() {
				assert_eq!(
					vec![
						Add { index: 6, char: 'x' },
						Add { index: 7, char: 'y' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobarxy"), None)
				)
			}
			#[test]
			fn eee() {
				assert_eq!(
					vec![
						Add { index: 6, char: 'x' },
						Add { index: 7, char: 'y' },
						Add { index: 8, char: 'z' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobarxyz"), None)
				)
			}
			#[test]
			fn m() {
				assert_eq!(
					vec![Add { index: 3, char: 'x' }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fooxbar"), None)
				)
			}
			#[test]
			fn mm() {
				assert_eq!(
					vec![
						Add { index: 3, char: 'x' },
						Add { index: 4, char: 'y' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fooxybar"), None)
				)
			}
			#[test]
			fn mmm() {
				assert_eq!(
					vec![
						Add { index: 3, char: 'x' },
						Add { index: 4, char: 'y' },
						Add { index: 5, char: 'z' },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fooxyzbar"), None)
				)
			}
		}

		#[cfg(feature="remove")]
		mod remove {
			use super::*;
			use Action::Remove;
			#[test]
			fn b() {
				assert_eq!(
					vec![Remove { index: 0 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("oobar"), None)
				)
			}
			#[ignore = "discard solves it better"]
			#[test]
			fn bb() {
				assert_eq!(
					vec![
						Remove { index: 0 },
						Remove { index: 0 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("obar"), None)
				)
			}
			#[ignore = "discard solves it better"]
			#[test]
			fn bbb() {
				assert_eq!(
					vec![
						Remove { index: 0 },
						Remove { index: 0 },
						Remove { index: 0 },
					],
					find_solution_st(WordEng::new("foobar"), WordEng::new("bar"), None)
				)
			}
			#[test]
			fn e() {
				assert_eq!(
					vec![Remove { index: 5 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fooba"), None)
				)
			}
			#[ignore = "discard solves it better"]
			#[test]
			fn ee() {
				let expected_solutions = vec![
					vec![
						Remove { index: 5 },
						Remove { index: 4 },
					],
					vec![
						Remove { index: 4 },
						Remove { index: 4 },
					],
				];
				let actual_solution = find_solution_st(WordEng::new("foobar"), WordEng::new("foob"), None);
				dbg!(&expected_solutions, &actual_solution);
				assert!(expected_solutions.contains(&actual_solution))
			}
			#[ignore = "discard solves it better"]
			#[test]
			fn eee() {
				let expected_solutions = vec![
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
				];
				let actual_solution = find_solution_st(WordEng::new("foobar"), WordEng::new("foo"), None);
				dbg!(&expected_solutions, &actual_solution);
				assert!(expected_solutions.contains(&actual_solution))
			}
		}

		#[cfg(feature="replace")]
		mod replace {
			use super::*;
			use Action::Replace;
			#[test]
			fn b() {
				assert_eq!(
					vec![Replace { index: 0, char: 'x' }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("xoobar"), None)
				)
			}
			#[test]
			fn e() {
				assert_eq!(
					vec![Replace { index: 5, char: 'x' }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobax"), None)
				)
			}
			#[test]
			fn m() {
				assert_eq!(
					vec![Replace { index: 2, char: 'x' }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foxbar"), None)
				)
			}
		}

		#[cfg(feature="swap")]
		mod swap {
			use super::*;
			use Action::Swap;
			#[test]
			fn foobar_barfoo() {
				assert_eq!(
					vec![Swap { index1s: 0, index1e: 2, index2s: 3, index2e: 5 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("barfoo"), None)
				)
			}
			#[test]
			fn abcfoodefbarxyz_abcbardeffooxyz() {
				assert_eq!(
					vec![Swap { index1s: 3, index1e: 5, index2s: 9, index2e: 11 }],
					find_solution_st(WordEng::new("abcfoodefbarxyz"), WordEng::new("abcbardeffooxyz"), None)
				)
			}
		}

		#[cfg(feature="discard")]
		mod discard {
			use super::*;
			use Action::Discard;
			#[test]
			fn b2() {
				assert_eq!(
					vec![Discard { index_start: 0, index_end: 1 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("obar"), None)
				)
			}
			#[test]
			fn b3() {
				assert_eq!(
					vec![Discard { index_start: 0, index_end: 2 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("bar"), None)
				)
			}
			#[test]
			fn e2() {
				assert_eq!(
					vec![Discard { index_start: 4, index_end: 5 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foob"), None)
				)
			}
			#[test]
			fn e3() {
				assert_eq!(
					vec![Discard { index_start: 3, index_end: 5 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foo"), None)
				)
			}
			#[test]
			fn m2() {
				assert_eq!(
					vec![Discard { index_start: 2, index_end: 3 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foar"), None)
				)
			}
			#[test]
			fn m4() {
				assert_eq!(
					vec![Discard { index_start: 1, index_end: 4 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("fr"), None)
				)
			}
		}

		#[cfg(feature="take")]
		mod take {
			use super::*;
			use Action::Take;
			#[test]
			fn foobarxyz_foo() {
				assert_eq!(
					vec![Take { index_start: 3, index_end: 5 }],
					find_solution_st(WordEng::new("foobarxyz"), WordEng::new("bar"), None)
				)
			}
		}

		#[cfg(feature="copy")]
		mod copy {
			use super::*;
			use Action::Copy_;
			#[test]
			fn abcd_ababcdcd() {
				assert_eq!(
					vec![Copy_ { index_start: 0, index_end: 3, index_insert: 2 }],
					find_solution_st(WordEng::new("abcd"), WordEng::new("ababcdcd"), None)
				)
			}
			#[test]
			fn foo_foofoo() {
				let expected_solutions = [
					vec![Copy_ { index_start: 0, index_end: 2, index_insert: 0 }],
					vec![Copy_ { index_start: 0, index_end: 2, index_insert: 3 }],
				];
				let actual_solution = find_solution_st(WordEng::new("foo"), WordEng::new("foofoo"), None);
				dbg!(&expected_solutions, &actual_solution);
				assert!(expected_solutions.contains(&actual_solution))
			}
			#[test]
			fn foobar_foobarfoo() {
				assert_eq!(
					vec![Copy_ { index_start: 0, index_end: 2, index_insert: 6 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("foobarfoo"), None)
				)
			}
			#[test]
			fn foobar_barfoobar() {
				assert_eq!(
					vec![Copy_ { index_start: 3, index_end: 5, index_insert: 0 }],
					find_solution_st(WordEng::new("foobar"), WordEng::new("barfoobar"), None)
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
			dbg!(expected_solutions, actual_solution);
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

	mod random_search {
		use super::*;
		mod stability {
			use super::*;
			mod from_same {
				use super::*;
				#[test]
				fn empty() {
					let word = WordEng::new("");
					for _ in 0..10_000 {
						let random_action = word.gen_random_legal_action();
						let _new_word = word.apply_action(random_action);
					}
				}
				#[test]
				fn a() {
					let word = WordEng::new("a");
					for _ in 0..10_000 {
						let random_action = word.gen_random_legal_action();
						let _new_word = word.apply_action(random_action);
					}
				}
				#[test]
				fn ab() {
					let word = WordEng::new("ab");
					for _ in 0..100_000 {
						let random_action = word.gen_random_legal_action();
						let _new_word = word.apply_action(random_action);
					}
				}
				#[test]
				fn abc() {
					let word = WordEng::new("abc");
					for _ in 0..1_000_000 {
						let random_action = word.gen_random_legal_action();
						let _new_word = word.apply_action(random_action);
					}
				}
			}
			mod chain {
				use super::*;
				#[test]
				fn empty() {
					let mut word = WordEng::new("");
					for _ in 0..10_000 {
						let random_action = word.gen_random_legal_action();
						word.apply_action_mut(random_action);
					}
				}
				#[test]
				fn a() {
					let mut word = WordEng::new("a");
					for _ in 0..10_000 {
						let random_action = word.gen_random_legal_action();
						word.apply_action_mut(random_action);
					}
				}
				#[test]
				fn ab() {
					let mut word = WordEng::new("ab");
					for _ in 0..10_000 {
						let random_action = word.gen_random_legal_action();
						word.apply_action_mut(random_action);
					}
				}
				#[test]
				fn abc() {
					let mut word = WordEng::new("abc");
					for _ in 0..10_000 {
						let random_action = word.gen_random_legal_action();
						word.apply_action_mut(random_action);
					}
				}
			}
		}
		mod completeness {
			use super::*;
			#[ignore = "TODO"]
			#[test]
			fn todo() {
				todo!()
			}
		}
	}
}

