import numpy as np


def GMF_model(x_data, rou, sigma, kg):

    # Unpack the angular geometry
    thetas = x_data[:, 0]  # Solar zenith angle
    thetav = x_data[:, 1]  # Viewing zenith angle
    fai = x_data[:, 2]     # Relative azimuth angle

    # Define the fixed refractive index
    n = 1.5

    # Calculate the cosine of the phase angle g
    cos_g = (
        np.cos(thetas) * np.cos(thetav)
        + np.sin(thetas) * np.sin(thetav) * np.cos(fai)
    )

    # Calculate the incidence angle on the specular facet
    thetai = np.arccos(cos_g) / 2.0

    # Calculate the refraction angle
    thetat = np.arcsin(np.sin(thetai) / n)

    # Calculate the two Fresnel reflection components
    Fq = 1 / 2 * (
        (n * np.cos(thetat) - np.cos(thetai))
        / (n * np.cos(thetat) + np.cos(thetai))
    ) ** 2

    Fh = 1 / 2 * (
        (n * np.cos(thetai) - np.cos(thetat))
        / (n * np.cos(thetai) + np.cos(thetat))
    ) ** 2

    # Calculate the polarized Fresnel reflection term
    F = Fq - Fh

    # Calculate the microfacet tilt angle beta
    beita = np.arccos(
        (np.cos(thetas) + np.cos(thetav))
        / (2 * np.cos(thetai))
    )

    # Calculate the GGX-based microfacet distribution function
    P = (
        sigma ** 2
        / (
            np.pi
            * (
                (np.cos(beita) ** 2) * (sigma ** 2 - 1)
                + 1
            ) ** 2
        )
    )

    # Calculate the shadowing function
    f = ((1 + np.cos(kg * 2 * thetai)) / 2) ** 3

    # Calculate the angular scaling term
    sc = (
        np.pi
        / (
            4
            * np.cos(beita)
            * (np.cos(thetas) + np.cos(thetav))
        )
    )

    # Calculate the bidirectional polarized reflectance factor
    BPRF = rou * sc * F * P * f

    return BPRF