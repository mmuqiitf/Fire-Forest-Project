import math
def bgr_to_hsv(b, g, r):
    b, g, r = r / 255.0, g / 255.0, b / 255.0
    print('r\' : ' + str(r))
    print('g\' : ' + str(g))
    print('b\' : ' + str(b))
    mx = max(r, g, b)
    mn = min(r, g, b)
    print('cmin : ' + str(mn))
    print('cmax : ' + str(mx))
    df = mx - mn
    print('delta : ' + str(df))
    if mx == mn:
        h = 0
        print('H : ' + str(h))
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
        print('H : ' + str(h))
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
        print('H : ' + str(h))
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
        print('H : ' + str(h))
    if mx == 0:
        s = 0
        print('s : ' + str(s))
    else:
        s = (df / mx)
        print('S : ' + str(s))
    v = mx
    print('V : ' + str(v))
    return h, s, v

def hsv_to_rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    print('Hi : ' + str(hi))
    f = h60 - h60f
    print('F : ' + str(f))
    p = v * (1 - s)
    print('P : ' + str(p))
    q = v * (1 - f * s)
    print('Q : ' + str(q))
    t = v * (1 - (1 - f) * s)
    print('T : ' + str(t))
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

print(14369.07264 % 360)
print(bgr_to_hsv(163, 180, 196))
# print(hsv_to_rgb(30.92736, 0.1683673469387756, 0.7686274509803922))
