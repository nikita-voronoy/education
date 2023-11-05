# Snake Game with AI

This project is a Rust-based implementation of the classic Snake game with an additional feature: an AI player. It employs `ggez` for rendering the game environment and `tch-rs` for interfacing with the Torch library, enabling the neural network to learn and adapt to the gameplay.

## Features

- **AI Learning**: Employs a neural network to learn the Snake game mechanics and improve over time.
- **Classic Snake Mechanics**: Includes all the traditional gameplay elements such as eating food to grow longer and avoiding collisions.
- **Graphics Rendering**: Utilizes the `ggez` crate for rendering graphics, providing a visually appealing gaming experience.
- **Adjustable Settings**: Allows for customization of grid size, tile size, update rate, and more.

## Installation

Before you begin, ensure that Rust and Cargo are installed on your machine. Refer to the [Rust Installation Guide](https://www.rust-lang.org/tools/install) for instructions.

To get started with this project:

1. Clone the repository:
   ```bash
   git clone https://nikita-voronoy/education.git
   cd education
   ```

2. Build the project using Cargo:
   ```bash
   cargo build --release
   ```

3. Run the game:
   ```bash
   cargo run
   ```

## Gameplay

- Use `W`, `A`, `S`, `D` or arrow keys for snake movement.
- Grow your snake by eating food, and increase your score with each item consumed.
- Avoid hitting the walls or the snake's own body.

## AI Gameplay

The game initiates with an AI-controlled snake by default. The AI decides the snake's moves based on the neural network's predictions.

## Development

If you're interested in contributing to the development of this Snake game or wish to customize it further:

1. The AI logic can be tweaked within the `ai` module.
2. Game settings can be modified in the `Game` struct.

Feel free to fork this project and submit pull requests with your improvements!

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
