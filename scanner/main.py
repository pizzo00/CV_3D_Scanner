from circular_marker import CircularMarker, MarkerColors

if __name__ == '__main__':
    circular_marker = CircularMarker()

    a = circular_marker.get_markers_position([MarkerColors.Black, MarkerColors.Magenta, MarkerColors.Magenta, MarkerColors.Cyan])
    b = circular_marker.get_markers_position([MarkerColors.Cyan, MarkerColors.Yellow, MarkerColors.White, MarkerColors.Magenta])
    c = circular_marker.get_markers_position([MarkerColors.Cyan, MarkerColors.Magenta, MarkerColors.Magenta, MarkerColors.Magenta])
    pass
