from matrx.objects.env_object import EnvObject

class ObstacleObject(EnvObject):
    def __init__(self, location, name, visualize_shape, img_name):
        """"
        An obstacle object not traversable, movable, and only visible when in front.
        """
        super().__init__(location, name, img_name, visualize_shape='img', visualize_size=1, class_callable=ObstacleObject, is_traversable=False, is_movable=True, is_collectable=True)