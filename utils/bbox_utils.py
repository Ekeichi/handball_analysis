def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    return (int(x), int(y))

def get_bbox_width(bbox):
    x1, y1, x2, y2 = bbox
    return x2 - x1
    
def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5 #euclidean distance
