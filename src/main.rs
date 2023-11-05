mod ai;

extern crate ggez;
extern crate rand;

use ggez::{Context, ContextBuilder, GameError, GameResult};
use ggez::event::{self, EventHandler, KeyCode, KeyMods};
use ggez::graphics::{self, Color, MeshBuilder};
use rand::Rng;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tch::{Device, Tensor};
use std::ops::Add;

const GRID_SIZE: (i16, i16) = (30, 30);
const TILE_SIZE: f32 = 20.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    fn to_point_delta(&self) -> Point {
        match self {
            Direction::Up => Point(0, -1),
            Direction::Down => Point(0, 1),
            Direction::Left => Point(-1, 0),
            Direction::Right => Point(1, 0),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct Point(i16, i16);

impl Add for Point {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Point(self.0 + other.0, self.1 + other.1)
    }
}

struct Game {
    snake: VecDeque<Point>,
    direction: Direction,
    food: Point,
    game_over: bool,
    score: u32,
    update_rate: Duration,
    last_update: Instant,
    ai: ai::AI,
}

impl Game {

    pub(crate) fn get_state_as_tensor(&self) -> Tensor {
        let mut grid = vec![0; (GRID_SIZE.0 * GRID_SIZE.1) as usize];
        for point in &self.snake {
            let idx = (point.0 + point.1 * GRID_SIZE.0) as usize;
            grid[idx] = 1;
        }

        let food_idx = (self.food.0 + self.food.1 * GRID_SIZE.0) as usize;
        grid[food_idx] = 2;

        Tensor::of_slice(&grid)
            .view((1, GRID_SIZE.1 as i64, GRID_SIZE.0 as i64))
    }

    fn new_food_location(snake: &VecDeque<Point>) -> Point {
        let mut rng = rand::thread_rng();
        loop {
            let new_point = Point(
                rng.gen_range(0..GRID_SIZE.0),
                rng.gen_range(0..GRID_SIZE.1),
            );
            if !snake.contains(&new_point) {
                return new_point;
            }
        }
    }

    fn action_to_direction(&self, action: i64) -> Direction {
        match action {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => panic!("Invalid action"),
        }
    }

    fn update_snake_position(&mut self) {
        let new_head = self.direction.to_point_delta() + *self.snake.front().expect("Snake has no body");
        self.snake.push_front(new_head);
        if new_head != self.food {
            self.snake.pop_back();
        } else {
            self.score += 1;
            self.food = Game::new_food_location(&self.snake);
        }
    }

    fn get_target_tensor_from_game_over(&self) -> Tensor {
        let mut target = vec![0; (GRID_SIZE.0 * GRID_SIZE.1) as usize];
        for point in &self.snake {
            let idx = (point.0 + point.1 * GRID_SIZE.0) as usize;
            target[idx] = 1;
        }
        Tensor::of_slice(&target)
            .view((1, GRID_SIZE.1 as i64, GRID_SIZE.0 as i64))
    }
}

impl EventHandler<GameError> for Game {
    fn update(&mut self, _ctx: &mut Context) -> GameResult<()> {
        if self.game_over {
            return Ok(());
        }

        if Instant::now() - self.last_update >= self.update_rate {
            self.last_update = Instant::now();

            let state = self.get_state_as_tensor();

            let action = self.ai.choose_action(&state);

            self.direction = self.action_to_direction(action);


            self.update_snake_position();

            if self.game_over {
                let target = self.get_target_tensor_from_game_over();
                self.ai.train(&state, &target);
            }
        }

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, Color::WHITE);

        if self.game_over {
            return graphics::present(ctx);
        }

        for point in &self.snake {
            let rectangle = MeshBuilder::new().rectangle(
                graphics::DrawMode::fill(),
                graphics::Rect::new_i32(
                    point.0 as i32 * TILE_SIZE as i32,
                    point.1 as i32 * TILE_SIZE as i32,
                    TILE_SIZE as i32,
                    TILE_SIZE as i32,
                ),
                Color::GREEN,
            )?.build(ctx)?;

            graphics::draw(ctx, &rectangle, graphics::DrawParam::default())?;
        }

        let food_rectangle = MeshBuilder::new().rectangle(
            graphics::DrawMode::fill(),
            graphics::Rect::new_i32(
                self.food.0 as i32 * TILE_SIZE as i32,
                self.food.1 as i32 * TILE_SIZE as i32,
                TILE_SIZE as i32,
                TILE_SIZE as i32,
            ),
            Color::RED,
        )?.build(ctx)?;

        graphics::draw(ctx, &food_rectangle, graphics::DrawParam::default())?;

        graphics::present(ctx)?;

        Ok(())
    }

    fn key_down_event(&mut self, _ctx: &mut Context, keycode: KeyCode, _keymod: KeyMods, _repeat: bool) {
        if self.game_over {
            return;
        }

        let new_direction = match keycode {
            KeyCode::W | KeyCode::Up if self.direction != Direction::Down => Direction::Up,
            KeyCode::A | KeyCode::Left if self.direction != Direction::Right => Direction::Left,
            KeyCode::S | KeyCode::Down if self.direction != Direction::Up => Direction::Down,
            KeyCode::D | KeyCode::Right if self.direction != Direction::Left => Direction::Right,
            _ => return,
        };
        self.direction = new_direction;
    }
}

fn main() -> GameResult {
    let (ctx, event_loop) = ContextBuilder::new("snake_game", "author")
        .build()
        .expect("failed to build context");

    let game = Game {
        snake: VecDeque::from(vec![Point(GRID_SIZE.0 / 2, GRID_SIZE.1 / 2)]),
        direction: Direction::Up,
        food: Game::new_food_location(&VecDeque::new()),
        game_over: false,
        score: 0,
        update_rate: Duration::from_millis(100),
        last_update: Instant::now(),
        ai: ai::AI::new(Device::Cuda(0)),
    };

    event::run(ctx, event_loop, game)
}
