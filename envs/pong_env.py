"""
This module gives a customised version of Pong that uses Pygame. It's made to be used with reinforcement learning to teach a computer how to play Pong.

Pong has a state space that includes the position and speed of the ball and paddles. The actions available are just moving the paddle up, down and stay sturdy.
"""
import random
import pygame

# define frame rate and window size for pygame to render
FPS = 60
WINDOW_SIZE = {"WIDTH": 400, "HEIGHT": 400}
WINDOW_MARGIN = 15

PADDLE_SIZE = {"WIDTH": 10, "HEIGHT": 60}

PADDLE_SPEED = 3

BALL_SIZE = {"WIDTH": 10, "HEIGHT": 10}

BALL_SPEED = {"X": 3, "Y": 2}


COLOURS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "RED": (255, 0, 0),
    "BLUE": (0, 0, 255),
    "YELLOW": (255, 255, 0),
    "GREEN": (0, 255, 0),
}

screen = pygame.display.set_mode((WINDOW_SIZE.get("WIDTH"), WINDOW_SIZE.get("HEIGHT")))


def render_ball(_x_ball, _y_ball, _colour_ball):
    """
    Renders the ball on the given screen at its current position.
    :param _x_ball:
    :param _y_ball:
    :param _colour_ball:
    :return: None
    """
    ball = pygame.Rect(
        _x_ball, _y_ball, BALL_SIZE.get("WIDTH"), BALL_SIZE.get("HEIGHT")
    )
    pygame.draw.rect(screen, _colour_ball, ball)


def render_our_paddle(_y_paddle):
    """
    Renders our paddle based on its vertical position.
    :param _y_paddle:
    :return: None
    """
    paddle = pygame.Rect(
        WINDOW_MARGIN, _y_paddle, PADDLE_SIZE.get("WIDTH"), PADDLE_SIZE.get("HEIGHT")
    )
    pygame.draw.rect(screen, COLOURS.get("YELLOW"), paddle)


def render_rival_paddle(_y_paddle):
    """
        Renders our rival's paddle based on its vertical position.
        :param _y_paddle:
        :return: None
        """
    paddle = pygame.Rect(
        WINDOW_SIZE.get("WIDTH") - WINDOW_MARGIN - PADDLE_SIZE.get("WIDTH"),
        _y_paddle,
        PADDLE_SIZE.get("WIDTH"),
        PADDLE_SIZE.get("HEIGHT"),
    )
    pygame.draw.rect(screen, COLOURS.get("WHITE"), paddle)


def update_observation(
    _y_our_paddle,
    _y_rival_paddle,
    _x_ball,
    _y_ball,
    _x_ball_direction,
    _y_ball_direction,
    _colour_ball,
    _d_frame_rate=7.5,
):
    """
    Updates the view of the environment as a result of the agent's action.
    :param _y_our_paddle:
    :param _y_rival_paddle:
    :param _x_ball:
    :param _y_ball:
    :param _x_ball_direction:
    :param _y_ball_direction:
    :param _colour_ball:
    :param _d_frame_rate:
    :return: observation as array
    """
    _x_ball = _x_ball + _x_ball_direction * BALL_SPEED.get("X") * _d_frame_rate
    _y_ball = _y_ball + _y_ball_direction * BALL_SPEED.get("Y") * _d_frame_rate
    score = 0
    updated_ball_colour = _colour_ball

    if (
        _x_ball <= WINDOW_MARGIN + PADDLE_SIZE.get("WIDTH")
        and _y_ball + BALL_SIZE.get("HEIGHT") >= _y_our_paddle
        and _y_ball - BALL_SIZE.get("HEIGHT")
        <= _y_our_paddle + PADDLE_SIZE.get("HEIGHT")
        and _x_ball_direction == -1
    ):
        _x_ball_direction = 1
        score = 10.0
        updated_ball_colour = COLOURS.get("BLUE")

    elif _x_ball <= 0:
        _x_ball_direction = 1
        score = -10.0
        updated_ball_colour = COLOURS.get("RED")
        return [
            score,
            _x_ball,
            _y_ball,
            _x_ball_direction,
            _y_ball_direction,
            updated_ball_colour,
        ]

    if (
        _x_ball >= WINDOW_SIZE.get("WIDTH") - PADDLE_SIZE.get("WIDTH") - WINDOW_MARGIN
        and _y_ball + BALL_SIZE.get("HEIGHT") >= _y_rival_paddle
        and _y_ball - BALL_SIZE.get("HEIGHT")
        <= _y_rival_paddle + PADDLE_SIZE.get("HEIGHT")
    ):
        _x_ball_direction = -1
        updated_ball_colour = COLOURS.get("WHITE")
    elif _x_ball >= WINDOW_SIZE.get("WIDTH") - BALL_SIZE.get("WIDTH"):
        _x_ball_direction = -1
        updated_ball_colour = COLOURS.get("WHITE")
        return [
            score,
            _x_ball,
            _y_ball,
            _x_ball_direction,
            _y_ball_direction,
            updated_ball_colour,
        ]

    if _y_ball <= 0:
        _y_ball = 0
        _y_ball_direction = 1
    elif _y_ball >= WINDOW_SIZE.get("HEIGHT") - BALL_SIZE.get("HEIGHT"):
        _y_ball = WINDOW_SIZE.get("HEIGHT") - BALL_SIZE.get("HEIGHT")
        _y_ball_direction = -1
    return [
        score,
        _x_ball,
        _y_ball,
        _x_ball_direction,
        _y_ball_direction,
        updated_ball_colour,
    ]


def update_our_position(action, _y_paddle, _d_frame_rate):
    """
    Update our paddle position based on action taken.
    :param action:
    :param _y_paddle:
    :param _d_frame_rate:
    :return: vertical position of our paddle
    """
    _d_frame_rate = 7.5
    if action == 1:
        _y_paddle = _y_paddle - PADDLE_SPEED * _d_frame_rate
    if action == 2:
        _y_paddle = _y_paddle + PADDLE_SPEED * _d_frame_rate
    _y_paddle = max(_y_paddle, 0)
    if _y_paddle > WINDOW_SIZE.get("HEIGHT") - PADDLE_SIZE.get("HEIGHT"):
        _y_paddle = WINDOW_SIZE.get("HEIGHT") - PADDLE_SIZE.get("HEIGHT")
    return _y_paddle


def update_rival_position(_y_paddle, _y_ball, _d_frame_rate):
    """
    Update our paddle position based on action taken.
    :param _y_paddle:
    :param _y_ball:
    :param _d_frame_rate:
    :return:
    """
    _d_frame_rate = 7.5
    if (
        _y_paddle + PADDLE_SIZE.get("HEIGHT") / 2
        < _y_ball + BALL_SIZE.get("HEIGHT") / 2
    ):
        _y_paddle = _y_paddle + PADDLE_SPEED * _d_frame_rate
    if (
        _y_paddle + PADDLE_SIZE.get("HEIGHT") / 2
        > _y_ball + BALL_SIZE.get("HEIGHT") / 2
    ):
        _y_paddle = _y_paddle - PADDLE_SPEED * _d_frame_rate
    _y_paddle = max(_y_paddle, 0)
    if _y_paddle > WINDOW_SIZE.get("HEIGHT") - PADDLE_SIZE.get("HEIGHT"):
        _y_paddle = WINDOW_SIZE.get("HEIGHT") - PADDLE_SIZE.get("HEIGHT")
    return _y_paddle


class PongGame:
    """
    Pong environment made for reinforcement learning agents.
    This class presents an customised version of the classic Pong game using the Pygame library.
    """
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Pong Environment")

        seed = random.randint(0, 9)

        # initialie positions of paddle
        self.y_our_paddle = (
            WINDOW_SIZE.get("HEIGHT") / 2 - PADDLE_SIZE.get("HEIGHT") / 2
        )
        self.y_rival_paddle = (
            WINDOW_SIZE.get("HEIGHT") / 2 - PADDLE_SIZE.get("HEIGHT") / 2
        )

        self.x_ball_direction = 1
        self.y_ball_direction = 1

        self.x_ball = WINDOW_SIZE.get("WIDTH") / 2 - BALL_SIZE.get("WIDTH") / 2

        self.clock = pygame.time.Clock()
        self.colour_ball = COLOURS.get("WHITE")

        self.frame_display_count = 0
        self.score_display = -10.0
        self.epsilon_display = 1.0

        self.font = pygame.font.SysFont("calibri", 20)

        if 0 < seed < 3:
            self.x_ball_direction = 1
            self.y_ball_direction = 1
        if 3 <= seed < 5:
            self.x_ball_direction = -1
            self.y_ball_direction = 1
        if 5 <= seed < 8:
            self.x_ball_direction = 1
            self.y_ball_direction = -1
        if 8 <= seed < 10:
            self.x_ball_direction = -1
            self.y_ball_direction = -1

        seed = random.randint(0, 9)

        self.y_pong = seed * (WINDOW_SIZE.get("HEIGHT") - BALL_SIZE.get("HEIGHT")) / 9

        # Initialise Game

    def init_render(self):
        """
        Guide Pygame to render components on guilded colours
        :return: None
        """
        pygame.event.pump()
        screen.fill(COLOURS.get("BLACK"))
        render_our_paddle(self.y_our_paddle)
        render_rival_paddle(self.y_rival_paddle)
        render_ball(self.x_ball, self.y_pong, COLOURS.get("WHITE"))
        pygame.display.flip()

    def take_action(self, action):
        """
        Change environment attributes based on action taken.
        :param action:
        :return: observation as array
        """
        delta_frame_time = self.clock.tick(FPS)
        pygame.event.pump()
        screen.fill(COLOURS.get("BLACK"))

        self.y_our_paddle = update_our_position(
            action, self.y_our_paddle, delta_frame_time
        )
        render_our_paddle(self.y_our_paddle)

        self.y_rival_paddle = update_rival_position(
            self.y_rival_paddle, self.y_pong, delta_frame_time
        )
        render_rival_paddle(self.y_rival_paddle)

        [
            score,
            self.x_ball,
            self.y_pong,
            self.x_ball_direction,
            self.y_ball_direction,
            self.colour_ball,
        ] = update_observation(
            self.y_our_paddle,
            self.y_rival_paddle,
            self.x_ball,
            self.y_pong,
            self.x_ball_direction,
            self.y_ball_direction,
            self.colour_ball
        )

        render_ball(self.x_ball, self.y_pong, self.colour_ball)

        if score > 0.5 or score < -0.5:
            self.score_display = 0.05 * score + self.score_display * 0.95

        _score_display = self.font.render(
            "Score: " + str("{0:.2f}".format(self.score_display)), True, (255, 255, 255)
        )
        screen.blit(_score_display, (50.0, 20.0))

        _time_display = self.font.render(
            "Time: " + str(self.frame_display_count), True, (255, 255, 255)
        )
        screen.blit(_time_display, (50.0, 40.0))

        _epsilon_display = self.font.render(
            "Ep: " + str("{0:.4f}".format(self.epsilon_display)), True, (255, 255, 255)
        )
        screen.blit(_epsilon_display, (50.0, 60.0))

        pygame.display.flip()

        return [
            score,
            self.y_our_paddle,
            self.x_ball,
            self.y_pong,
            self.x_ball_direction,
            self.y_ball_direction,
        ]

    def get_current_state(self):
        """
        :return: current environment state as array
        """
        return [
            self.y_our_paddle,
            self.x_ball,
            self.y_pong,
            self.x_ball_direction,
            self.y_ball_direction,
        ]

    def re_render_display(self, _time, epsilon):
        """
        Update frame count and epsilon to display
        :param _time:
        :param epsilon:
        :return: None
        """
        self.frame_display_count = _time
        self.epsilon_display = epsilon
