//! Build script for shgo-rs
//! 
//! Copies pre-written C/C++ header files to the include directory

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let include_dir = PathBuf::from(&crate_dir).join("include");

    // Create the include directory if it doesn't exist
    std::fs::create_dir_all(&include_dir).unwrap();

    // The headers are manually written and stored in the include directory
    // This is more reliable than cbindgen for complex crates
    
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=include/shgo.h");
    println!("cargo:rerun-if-changed=include/shgo.hpp");
}
