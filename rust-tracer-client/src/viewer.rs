//! The window: assembles the blocks reported by the runs and forwards the
//! user's keys as commands. Runs on the main thread (SDL requires it); the
//! runs report through the SDL event queue, as [`RunEvent`] custom events.

use std::time::{Duration, Instant};

use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use tokio::sync::mpsc::UnboundedSender;

use crate::blocks::{blit, outline_color};
use crate::demo::{Command, RunEvent};

/// Window title state.
#[derive(Debug, Default)]
struct Status {
    done: usize,
    total: usize,
    paused: bool,
    finished: Option<Option<String>>,
}

impl Status {
    fn title(&self) -> String {
        let mut title = format!("PiTracer on ArmoniK - {}/{}", self.done, self.total);
        match &self.finished {
            Some(None) => title.push_str(" - done"),
            Some(Some(error)) => title.push_str(&format!(" - failed: {error}")),
            None if self.paused => title.push_str(" (paused)"),
            None => (),
        }
        title
    }
}

/// Shows the image until the window is closed (or Esc). Keys: Space pauses
/// or resumes the session, R starts a new run, F11 toggles fullscreen.
/// Blocks still being refined are framed with a color going from blue to
/// red as their sample count goes from `samples.0` (one pass) to `samples.1`
/// (the cap).
pub fn run(
    sdl: &sdl2::Sdl,
    width: u32,
    height: u32,
    samples: (u32, u32),
    commands: &UnboundedSender<Command>,
) -> eyre::Result<()> {
    let video = sdl.video().map_err(eyre::Report::msg)?;
    let window = video
        .window("PiTracer on ArmoniK", width, height)
        .position_centered()
        .resizable()
        .build()?;
    let mut canvas = window.into_canvas().build()?;
    let texture_creator = canvas.texture_creator();
    let mut event_pump = sdl.event_pump().map_err(eyre::Report::msg)?;

    let pitch = width as usize * 3;
    let mut pixels = vec![0u8; pitch * height as usize];
    let mut status = Status::default();

    const PRESENT_INTERVAL: Duration = Duration::from_millis(15);
    let mut dirty = true;
    let mut title_dirty = true;
    let mut last_present = Instant::now() - PRESENT_INTERVAL;

    'running: loop {
        // Block until something happens, then drain the whole queue so a
        // burst of blocks is coalesced into a single present.
        let mut pending = Vec::new();
        if let Some(event) = event_pump.wait_event_timeout(100) {
            pending.push(event);
            pending.extend(event_pump.poll_iter());
        }

        for event in pending {
            // Single registered custom type: extract at most once per event
            // (the payload is moved out of a raw box; never clone user events).
            if let Some(event) = event.as_user_event_type::<RunEvent>() {
                match event {
                    RunEvent::Reset { session_id } => {
                        tracing::debug!(session_id, "new run");
                        pixels.fill(0);
                        status = Status::default();
                    }
                    RunEvent::Block { image, done } => {
                        let outline =
                            (!done).then(|| outline_color(image.samples, samples.0, samples.1));
                        blit(&mut pixels, width, &image, outline);
                    }
                    RunEvent::Progress { done, total } => {
                        status.done = done;
                        status.total = total;
                    }
                    RunEvent::Finished { error } => status.finished = Some(error),
                }
                dirty = true;
                title_dirty = true;
                continue;
            }
            match event {
                Event::Window {
                    win_event:
                        WindowEvent::Maximized
                        | WindowEvent::Minimized
                        | WindowEvent::Resized(..)
                        | WindowEvent::SizeChanged(..)
                        | WindowEvent::Restored,
                    ..
                } => dirty = true,
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => {
                    tracing::info!("stopping requested");
                    break 'running;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Space),
                    ..
                } if status.finished.is_none() => {
                    status.paused = !status.paused;
                    title_dirty = true;
                    _ = commands.send(Command::TogglePause);
                }
                Event::KeyDown {
                    keycode: Some(Keycode::R),
                    ..
                } => _ = commands.send(Command::Rerun),
                Event::KeyDown {
                    keycode: Some(Keycode::F11),
                    ..
                } => {
                    let window = canvas.window_mut();
                    let state = match window.fullscreen_state() {
                        sdl2::video::FullscreenType::Off => sdl2::video::FullscreenType::True,
                        _ => sdl2::video::FullscreenType::Off,
                    };
                    window.set_fullscreen(state).map_err(eyre::Report::msg)?;
                    dirty = true;
                }
                _ => {}
            }
        }

        if title_dirty {
            canvas
                .window_mut()
                .set_title(&status.title())
                .map_err(eyre::Report::msg)?;
            title_dirty = false;
        }

        if dirty && last_present.elapsed() >= PRESENT_INTERVAL {
            let surface = sdl2::surface::Surface::from_data(
                &mut pixels,
                width,
                height,
                pitch as u32,
                sdl2::pixels::PixelFormatEnum::RGB24,
            )
            .map_err(eyre::Report::msg)?;
            let texture = surface
                .as_texture(&texture_creator)
                .map_err(eyre::Report::msg)?;

            // Letterbox to keep the aspect ratio.
            let (w, h) = canvas.output_size().map_err(eyre::Report::msg)?;
            let target = match (width * h).cmp(&(w * height)) {
                std::cmp::Ordering::Greater => {
                    let sh = height * w / width;
                    Some(Rect::new(0, ((h - sh) / 2) as i32, w, sh))
                }
                std::cmp::Ordering::Less => {
                    let sw = width * h / height;
                    Some(Rect::new(((w - sw) / 2) as i32, 0, sw, h))
                }
                std::cmp::Ordering::Equal => None,
            };

            canvas.clear();
            canvas
                .copy(&texture, None, target)
                .map_err(eyre::Report::msg)?;
            canvas.present();

            dirty = false;
            last_present = Instant::now();
        }
    }
    Ok(())
}
