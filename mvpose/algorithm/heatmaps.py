from skimage.feature import peak_local_max


def get_all_peaks(heatmap, threshold=0.1):
    """
    extracts peaks from the heatmap
    :param heatmap:
    :param threshold:
    :return: [
            [ (x,y,score), (x,y,score),... ]  # Nose
            ...
    ]
    """
    peaks = []
    for i in range(18):
        hm = heatmap[:,:,i]
        local_peaks = peak_local_max(hm, threshold_abs=threshold)
        found_peaks = []
        for x,y in local_peaks:
            found_peaks.append((y,x,hm[x,y]))
        peaks.append(found_peaks)
    return peaks
