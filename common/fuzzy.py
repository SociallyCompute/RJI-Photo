import numpy as np

def generate_triangle(value, vals):
    a, b, c = vals[0], vals[1], vals[2]
    y = []
    for item in value:
        if item <= a:
            y.append(0)
        elif item >= a and item <= b:
            y.append((item-a)/(b-a))
        elif item >= b and item <= c:
            y.append((c-item)/(c-b))
        else:
            y.append(0)
    return np.array(y)

def generate_trapezoid(value, vals):
    a, b, c, d = vals[0], vals[1], vals[2], vals[3]
    y = []
    for item in value:
        if item < a:
            y.append(0)
        elif item >= a and item <= b:
            y.append((item-a)/(b-a))
        elif item >= b and item <= c:
            y.append(1)
        elif item >= c and item <= d:
            y.append((d-item)/(d-c))
        else:
            y.append(0)
    return np.array(y)

def generate_membership_funcs(func, vals):    
    '''
    assuming there are 3 measures to classify (Low, Medium, High)
    '''
    edges_qual = np.arange(1, 11, dtype=np.float32)
    color_qual = np.arange(1, 11, dtype=np.float32)
    total_qual = np.arange(1, 11, dtype=np.float32)

    combo_edges_quality = [func(edges_qual, vals[0][0]), func(edges_qual, vals[0][1]), func(edges_qual, vals[0][2])]
    combo_color_quality = [func(color_qual, vals[1][0]), func(color_qual, vals[1][1]), func(color_qual, vals[1][2])]
    combo_total_quality = [func(total_qual, vals[2][0]), func(total_qual, vals[2][1]), func(total_qual, vals[2][2])]
    
    return combo_edges_quality, combo_color_quality, combo_total_quality