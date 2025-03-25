import pygame
import pygame.freetype

from config import BLACK, RED, GREEN, YELLOW

SCORE_AREA_HEIGHT = 50


class SnakeRenderer:
    def __init__(self, grid_size, cell_size, render_rate):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_rate = render_rate
        self.display_width, self.display_height = grid_size * cell_size, grid_size * cell_size + SCORE_AREA_HEIGHT
        self.display = pygame.display.set_mode((self.display_width, self.display_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.freetype.SysFont("serif", 20)

    def render(self, state, score):
        self.display.fill(BLACK)

        if state.shape != (self.grid_size, self.grid_size):
            raise ValueError("Invalid state object to render")

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if state[y, x] == 1:
                    pygame.draw.rect(self.display, GREEN, self.get_rect(x, y))
                if state[y, x] == 2:
                    pygame.draw.rect(self.display, YELLOW, self.get_rect(x, y))
                elif state[y, x] == 3:
                    pygame.draw.rect(self.display, RED, self.get_rect(x, y))

        score_rect = pygame.Rect(0, self.display_height - SCORE_AREA_HEIGHT, self.display_width, SCORE_AREA_HEIGHT)
        pygame.draw.rect(self.display, (255, 255, 255), score_rect)

        score_text = self.font.render(f"Score: {score}", (0, 0, 0))
        self.display.blit(score_text[0], (10, self.grid_size * self.cell_size + 5))

        pygame.display.flip()
        self.clock.tick(self.render_rate)

    def close(self):
        pygame.quit()

    def get_rect(self, start_row, start_col):
        return start_row * self.cell_size, start_col * self.cell_size, self.cell_size, self.cell_size
