import pygame
import pandas as pd
import time
import sys


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode([1500, 500])

        self.screen_rect = self.screen.get_rect()
        self.mid_x = self.screen_rect.centerx
        self.mid_y = self.screen_rect.centery
        self.zoom = 5
        self.timestep = 0.1
        self.interval = 100

        self.font = pygame.font.SysFont('Verdana', 10)

        self.pos_x = 0
        self.t0 = time.time() - timesteps[0]

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys; sys.exit(0)

            if event.type == pygame.KEYDOWN:  # or event.type == pygame.KEYUP:
                print('pressed key', event.key)
                if (not (event.mod & pygame.KMOD_SHIFT)) and (not (event.mod & pygame.KMOD_CTRL)):
                    if event.key == 1073741903:
                        self.timestep += 0.1  # right
                    elif event.key == 1073741904:
                        self.timestep -= 0.1  # left
                    self.timestep = min(max(self.timestep, 0.0), 2.0)
                elif event.mod & pygame.KMOD_SHIFT:
                    if event.key == 1073741903:
                        self.zoom += 1  # right
                    elif event.key == 1073741904:
                        self.zoom -= 1  # left
                    self.zoom = min(max(self.zoom, 1), 10)
                elif event.mod & pygame.KMOD_CTRL:
                    if event.key == 1073741903:
                        self.interval += 10  # right
                    elif event.key == 1073741904:
                        self.interval -= 10  # left
                    self.interval = min(max(self.interval, 20), 1000)

    def step(self, time, car_types, car_positions, car_speeds):
        self.time = time
        self.car_types = car_types
        self.car_positions = car_positions
        self.car_speeds = car_speeds

        self.handle_events()

    def render(self):
        self.screen.fill((255, 255, 255))

        img = self.font.render(f'Timestep: {round(self.timestep, 1)}s (commands: Left / Right) (0 = real time)', True, (0, 0, 0))
        self.screen.blit(img, (20, 50))
        img = self.font.render(f'Zoom: x{int(self.zoom)} (commands: Shift + Left / Right)', True, (0, 0, 0))
        self.screen.blit(img, (20, 70))
        img = self.font.render(f'Interval: {int(self.interval)}m (commands: Ctrl + Left / Right)', True, (0, 0, 0))
        self.screen.blit(img, (20, 90))

        img = self.font.render(f'Time: {round(self.time, 1)}s', True, (0, 0, 0))
        self.screen.blit(img, (20, 130))
        img = self.font.render(f'Distance: {round(max(self.car_positions), 1)}m', True, (0, 0, 0))
        self.screen.blit(img, (20, 150))

        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(0, self.mid_y - 10 * self.zoom, self.screen_rect.width, 2 * 10 * self.zoom))

        shift_x = 0
        for car_type, car_x in zip(self.car_types, self.car_positions):
            if 'leader' in car_type or 'keyboard' in car_type:
                shift_x = self.mid_x - car_x * self.zoom

        for pos_x in range(-1000, 22000, self.interval):
            img = self.font.render(f'{self.interval}m', True, (0, 100, 0))
            self.screen.blit(img, ((pos_x + self.interval // 2) * self.zoom + shift_x - img.get_width() // 2, 10 - img.get_height() // 2))
            pygame.draw.rect(self.screen, (0, 200, 0), pygame.Rect(pos_x * self.zoom - 1 + shift_x, 0, 2, 20))

            img = self.font.render(f'{self.interval}m', True, (0, 100, 0))
            self.screen.blit(img, ((pos_x + self.interval // 2) * self.zoom + shift_x - img.get_width() //
                        2, self.screen_rect.height - 10 - img.get_height() // 2))
            pygame.draw.rect(self.screen, (0, 200, 0), pygame.Rect(pos_x * self.zoom - 1 + shift_x, self.screen_rect.height - 20, 2, 20))

        for car_type, car_x, car_speed in zip(self.car_types, self.car_positions, self.car_speeds):
            pos_x = car_x * self.zoom + shift_x
            pos_y = self.mid_y
            radius = 3 * self.zoom

            car_type_display = car_type
            if 'leader' in car_type:
                car_color = (0, 0, 255)
                car_type_display = 'Trajectory leader'
            elif 'keyboard' in car_type:
                car_color = (0, 255, 0)
                car_type_display = 'Keyboard'
            elif 'human' in car_type:
                car_color = (255, 255, 255)
                car_type_display = 'IDM'
            elif 'av' in car_type:
                car_color = (255, 0, 0)
                car_type_display = 'AV'

            pygame.draw.circle(self.screen, car_color, (pos_x, pos_y), radius)

            img = self.font.render(car_type_display, True, (0, 0, 0))
            self.screen.blit(img, (pos_x - img.get_width() // 2, pos_y - 15 * self.zoom - img.get_height() // 2))

            img = self.font.render(f'{round(car_speed, 1)}' + (' m/s' if 'leader' in car_type else ''), True, (0, 0, 0))
            self.screen.blit(img, (pos_x - img.get_width() // 2, pos_y + 15 * self.zoom - img.get_height() // 2))
            img = self.font.render(f'{round(car_speed * 3.6, 1)}' + (' km/h' if 'leader' in car_type else ''), True, (0, 0, 0))
            self.screen.blit(img, (pos_x - img.get_width() // 2, pos_y + 15 * self.zoom + 15 - img.get_height() // 2))
            img = self.font.render(f'{round(car_speed * 2.237, 1)}' + (' mph' if 'leader' in car_type else ''), True, (0, 0, 0))
            self.screen.blit(img, (pos_x - img.get_width() // 2, pos_y + 15 * self.zoom + 30 - img.get_height() // 2))

        pygame.display.flip()

        t = self.time
        if self.timestep < .05:
            time.sleep(max(self.t0 + t - time.time(), 0))
        else:
            self.t0 = time.time() - t
            if int(t * 10) % int(self.timestep * 10) != 0:
                pass


if __name__ == '__main__':
    print('Reading emissions data')

    emissions_path = sys.argv[1]
    df = pd.read_csv(emissions_path)
    #  time,step,id,position,speed,accel,headway,leader_speed,speed_difference,leader_id,follower_id,instant_energy_consumption,total_energy_consumption,total_distance_traveled,total_miles,total_gallons,avg_mpg

    timesteps = sorted(list(set(map(lambda x: round(x, 1), df['time']))))

    car_types = []
    car_positions = []
    car_speeds = []

    for ts in timesteps:
        ts_data = df.loc[df['time'] == ts]
        car_types.append(list(ts_data['id']))
        car_positions.append(list(ts_data['position']))
        car_speeds.append(list(ts_data['speed']))

    renderer = Renderer()
    for i in range(len(timesteps)):
        renderer.step(timesteps[i], car_types[i], car_positions[i], car_speeds[i])
        renderer.render()