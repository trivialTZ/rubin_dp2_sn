import requests
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from io import BytesIO

def plot_legacy_cutout(ra, dec, size=256, pixscale=0.262, layer="ls-dr10", fmt="jpg", ax=None):
    url = f"https://www.legacysurvey.org/viewer/cutout.{fmt}"
    params = dict(ra=ra, dec=dec, layer=layer, pixscale=pixscale, size=size)
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    if fmt == "jpg":
        img = plt.imread(BytesIO(r.content), format="jpg")
        ax.imshow(img, origin="lower")
        data = img
    else:  # fits, grz
        hdul = fits.open(BytesIO(r.content))
        d = hdul[0].data
        g, r_, z = d[0], d[1], d[2]
        rgb = make_lupton_rgb(z, r_, g, stretch=0.5, Q=8)
        ax.imshow(rgb, origin="lower")
        data = d

    ax.set_title(f"RA={ra:.5f}, Dec={dec:.5f}")
    ax.axis("off")
    return ax, data

if __name__ == "__main__":
    plot_legacy_cutout(150.1163, 2.2058, size=256, pixscale=0.262, layer="ls-dr10", fmt="jpg")