
class WallMarker:
    h = 230
    w = 130
    h_dec = 230.0
    w_dec = 130.0

    # points ordered [TopLeft, TopRight, BottLeft, BottRight]
    # world zero on top left
    points = [
        [0.0, 0.0, 0.0],
        [w_dec, 0.0, 0.0],
        [0.0, h_dec, 0.0],
        [w_dec, h_dec, 0.0],
    ]