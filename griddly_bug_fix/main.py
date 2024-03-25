import pickle
from pathlib import Path

import pygame
from griddly import gd
from griddly.gym import GymWrapper
from gymnasium.utils.play import display_arr
from pygame.locals import VIDEORESIZE

# Actions:
NOOP = [0, 0]
MOVE_LEFT = [0, 1]
MOVE_UP = [0, 2]
MOVE_RIGHT = [0, 3]
MOVE_DOWN = [0, 4]

keys = {
    (pygame.K_a,): MOVE_LEFT,
    (pygame.K_w,): MOVE_UP,
    (pygame.K_d,): MOVE_RIGHT,
    (pygame.K_s,): MOVE_DOWN,
}


def play(env, fps=10, keys_to_action=keys, noop=NOOP):
    env.reset()
    rendered = env.render_observer(render_mode='rgb_array')

    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            raise AssertionError(
                env.spec.id
                + ' does not have explicit key to action mapping, '
                + 'please specify one manually'
            )
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    video_size = [rendered.shape[1], rendered.shape[0]]

    pressed_keys = []
    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()
    while running:
        if env_done:
            env_done = False
            obs = env.reset()
        else:
            action = keys_to_action.get(tuple(sorted(pressed_keys)), noop)
            obs, rew, terminated, truncated, info = env.step(action)
            env_done = terminated or truncated
        if obs is not None:
            rendered = env.render_observer(render_mode='rgb_array')
            display_arr(screen, rendered, transpose=True, video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
                elif event.key == pygame.K_r:
                    state = env.get_state()
                    # with open('proximity_state.pkl', 'wb') as f:
                    #     pickle.dump(state, f)
                    # with open('proximity_state.pkl', 'rb') as f:
                    #     loaded_state = pickle.load(f)
                    #     assert state == loaded_state
                    env = env.load_state(state)
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


if __name__ == '__main__':
    # env_variant = 'proximity.yaml'
    env_variant = 'proximity_wo_healthbars.yaml'
    env = GymWrapper(
        str(Path(__file__).parent / env_variant),
        player_observer_type=gd.ObserverType.ISOMETRIC,
        global_observer_type=gd.ObserverType.ISOMETRIC,
        level=0,
    )

    play(env)
