def sub_sample(xs, duration, fps):
    nb_frames = len(xs)
    nb_subframes = int(duration*fps)
    if nb_frames<nb_subframes:
        return xs
    else:
        step = nb_frames//nb_subframes
        xs_sub = [xs[i] for i in range(0,nb_frames, step)]
        return xs_sub
