#constants
height = 12 #inches
diam = .87 #inches
rad = diam/2
origin = [0,0, height/2]

#formulas - use std form of the circle eqn
def is_inside_cyl(point, height, rad): #returns a boolean confirming whether it is in the cylinder or not
    if point[2]>(height/2) or point[2]<(-height/2):
        return False

    distance = math.sqrt((x - a)**2 + (y - b)**2)

    if distance != rad:
        return False
    
    return True
    
def closest_point_on_cylinder(point, height, rad, origin):
    if point[2]>(height/2):
        z = height/2
    elif point[2]<(-height/2):
        z = height/2
    else:
        z = point[2]

    x = point[0]/math.sqrt(point[0]**2 + point[1]**2)
    y = point[1]/math.sqrt(point[0]**2 + point[1]**2)
    x = x * rad
    y = y * rad
    return [x,y,z]
