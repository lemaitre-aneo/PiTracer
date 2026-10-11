//! Text in SVG images (the `svg-text` feature): fonts for usvg to lay text
//! out with, when the scene is loaded and the SVG made self-contained (see
//! `svg::self_contained`, which writes the text back as paths). What the
//! scene then carries, and what workers parse, is text-free: they need
//! neither fonts nor this feature, and render the same text.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use usvg::fontdb::Database;

use super::svg::{Missing, SvgError};

/// Sets up `options` to lay text out with the font `files` and, if
/// `system`, the fonts installed on the system. Fonts and characters no font
/// provides are recorded in `missing` (usvg would only log them, and draw
/// nothing or a placeholder glyph).
pub(super) fn configure(
    options: &mut usvg::Options,
    files: &[PathBuf],
    system: bool,
    missing: &Missing,
) -> Result<(), SvgError> {
    let mut database = Database::new();
    for file in files {
        load(&mut database, file)?;
    }
    // Generic families (`sans-serif`, ...) resolve to the first listed font,
    // so listed fonts take precedence over the system's.
    let first = database
        .faces()
        .next()
        .map(|face| face.families[0].0.clone());
    if let Some(family) = first {
        database.set_serif_family(family.clone());
        database.set_sans_serif_family(family.clone());
        database.set_monospace_family(family.clone());
        database.set_cursive_family(family.clone());
        database.set_fantasy_family(family);
    }
    if system {
        database.load_system_fonts();
    }

    let select_font = usvg::FontResolver::default_font_selector();
    let select_fallback = usvg::FontResolver::default_fallback_selector();
    options.font_resolver = usvg::FontResolver {
        select_font: Box::new({
            let missing = missing.clone();
            move |font, database| {
                let id = select_font(font, database);
                if id.is_none() {
                    let families: Vec<String> =
                        font.families().iter().map(ToString::to_string).collect();
                    missing.push(format!("font {}", families.join(", ")));
                }
                id
            }
        }),
        select_fallback: Box::new({
            let missing = missing.clone();
            move |character, used, database| {
                let id = select_fallback(character, used, database);
                if id.is_none() && !character.is_whitespace() && !character.is_control() {
                    missing.push(format!("character {character:?}"));
                }
                id
            }
        }),
    };
    options.fontdb = Arc::new(database);
    Ok(())
}

fn load(database: &mut Database, file: &Path) -> Result<(), SvgError> {
    let faces = database.len();
    database
        .load_font_file(file)
        .map_err(|source| SvgError::FontFile {
            path: file.to_owned(),
            source,
        })?;
    if database.len() == faces {
        return Err(SvgError::FontFile {
            path: file.to_owned(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, "not a font file"),
        });
    }
    Ok(())
}
