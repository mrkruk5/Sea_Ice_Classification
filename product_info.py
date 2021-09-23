from xml.etree import ElementTree
import numpy as np


class GeoExtentError(Exception):
    def __init__(self, message):
        self.message = message


def get_noise(path):
    # Extract noise floor information
    tree = ElementTree.parse(path)
    root = tree.getroot()
    nsmap = {'schemas': 'http://www.rsi.ca/rs2/prod/xml/schemas'}
    noise_e = root.findall(
        './/schemas:referenceNoiseLevel'
        '[@incidenceAngleCorrection="Sigma Nought"]',
        nsmap)[0]
    noise_val_e = noise_e.find('./schemas:noiseLevelValues', nsmap)
    noise = np.array([np.float64(val) for val in noise_val_e.text.split()])
    # plt.plot(noise)
    # plt.title('Sigma Nought Noise Level Values from product.xml')
    # plt.ylabel('dB')
    # plt.show()
    return noise


def get_look_up_table(path):
    tree = ElementTree.parse(path)
    root = tree.getroot()
    gains_e = root.find('./gains')
    gains = np.array([np.float64(val) for val in gains_e.text.split()])
    return gains


def get_dims(path):
    tree = ElementTree.parse(path)
    root = tree.getroot()
    nsmap = {'schemas': 'http://www.rsi.ca/rs2/prod/xml/schemas'}
    ras_attrs_e = root.findall('.//schemas:rasterAttributes', nsmap)[0]
    h = np.float64(ras_attrs_e.find('./schemas:numberOfLines', nsmap).text)
    w = np.float64(ras_attrs_e.find('./schemas:numberOfSamplesPerLine',
                                    nsmap).text)
    return h, w


def get_extents(path, height, width) -> list:
    idx_ext = [[0, 0], [height - 1, 0], [0, width - 1], [height - 1, width - 1]]
    geo_ext = np.empty(np.array(idx_ext).shape)
    corner_count = 0

    # Parse product.xml given in path to find the geo extents of img.
    tree = ElementTree.parse(path)
    root = tree.getroot()
    nsmap = {'schemas': 'http://www.rsi.ca/rs2/prod/xml/schemas'}
    geo_loc_grid_e = root.findall('.//schemas:geolocationGrid', nsmap)[0]
    for tie_point_e in geo_loc_grid_e:
        img_coord = tie_point_e.find('./schemas:imageCoordinate', nsmap)
        x_idx = np.float64(img_coord.find('./schemas:line', nsmap).text)
        y_idx = np.float64(img_coord.find('./schemas:pixel', nsmap).text)
        if [x_idx, y_idx] in idx_ext:
            geo_coord = tie_point_e.find('./schemas:geodeticCoordinate', nsmap)
            lat = np.float64(geo_coord.find('./schemas:latitude', nsmap).text)
            lon = np.float64(geo_coord.find('./schemas:longitude', nsmap).text)
            # Finding the index of [x_idx, y_idx] in idx_ext is needed
            # because the order of the geodetic coordinates in the
            # product.xml is not the same as the order with which GDAL prints
            # the extents.
            geo_ext[idx_ext.index([x_idx, y_idx])] = [lon, lat]
            corner_count += 1

    if corner_count != 4:
        raise GeoExtentError('The 4 geo-extents of the image were not found.')
    geo_ext = geo_ext.tolist()
    return geo_ext


def get_geo_height(path):
    # Parse product.xml given in path to find the height of the img.
    tree = ElementTree.parse(path)
    root = tree.getroot()
    nsmap = {'schemas': 'http://www.rsi.ca/rs2/prod/xml/schemas'}
    geo_loc_grid_e = root.findall('.//schemas:geolocationGrid', nsmap)[0]
    heights = []
    for tie_point_e in geo_loc_grid_e:
        geo_coord = tie_point_e.find('./schemas:geodeticCoordinate', nsmap)
        h = np.float64(geo_coord.find('./schemas:height', nsmap).text)
        heights.append(h)
    return np.mean(heights)


def get_rational_function(path):
    tree = ElementTree.parse(path)
    root = tree.getroot()
    nsmap = {'schemas': 'http://www.rsi.ca/rs2/prod/xml/schemas'}
    rf_e = root.findall('.//schemas:rationalFunctions', nsmap)[0]
    l_off = np.float64(rf_e.find('./schemas:lineOffset', nsmap).text)
    p_off = np.float64(rf_e.find('./schemas:pixelOffset', nsmap).text)
    lat_off = np.float64(rf_e.find('./schemas:latitudeOffset', nsmap).text)
    lon_off = np.float64(rf_e.find('./schemas:longitudeOffset', nsmap).text)
    h_off = np.float64(rf_e.find('./schemas:heightOffset', nsmap).text)
    l_scale = np.float64(rf_e.find('./schemas:lineScale', nsmap).text)
    p_scale = np.float64(rf_e.find('./schemas:pixelScale', nsmap).text)
    lat_scale = np.float64(rf_e.find('./schemas:latitudeScale', nsmap).text)
    lon_scale = np.float64(rf_e.find('./schemas:longitudeScale', nsmap).text)
    h_scale = np.float64(rf_e.find('./schemas:heightScale', nsmap).text)
    l_num_c_e = rf_e.find('./schemas:lineNumeratorCoefficients', nsmap)
    l_den_c_e = rf_e.find('./schemas:lineDenominatorCoefficients', nsmap)
    p_num_c_e = rf_e.find('./schemas:pixelNumeratorCoefficients', nsmap)
    p_den_c_e = rf_e.find('./schemas:pixelDenominatorCoefficients', nsmap)
    l_num_c = [np.float64(c) for c in l_num_c_e.text.split()]
    l_den_c = [np.float64(c) for c in l_den_c_e.text.split()]
    p_num_c = [np.float64(c) for c in p_num_c_e.text.split()]
    p_den_c = [np.float64(c) for c in p_den_c_e.text.split()]
    RF = RationalFunction(
        l_off=l_off,
        p_off=p_off,
        lat_off=lat_off,
        lon_off=lon_off,
        h_off=h_off,
        l_scale=l_scale,
        p_scale=p_scale,
        lat_scale=lat_scale,
        lon_scale=lon_scale,
        h_scale=h_scale,
        l_num_c=l_num_c,
        l_den_c=l_den_c,
        p_num_c=p_num_c,
        p_den_c=p_den_c,
    )
    return RF


class RationalFunction:
    def __init__(self, **kwargs):
        self.l_off = kwargs['l_off']
        self.p_off = kwargs['p_off']
        self.lat_off = kwargs['lat_off']
        self.lon_off = kwargs['lon_off']
        self.h_off = kwargs['h_off']
        self.l_scale = kwargs['l_scale']
        self.p_scale = kwargs['p_scale']
        self.lat_scale = kwargs['lat_scale']
        self.lon_scale = kwargs['lon_scale']
        self.h_scale = kwargs['h_scale']
        self.l_num_c = kwargs['l_num_c']
        self.l_den_c = kwargs['l_den_c']
        self.p_num_c = kwargs['p_num_c']
        self.p_den_c = kwargs['p_den_c']

    def lat_lon_index(self, lat, lon, h):
        # Todo: Add if/else condition for calculation of A when image
        #       straddles +/-180 degrees longitude.
        A = (lon - self.lon_off) / self.lon_scale
        B = (lat - self.lat_off) / self.lat_scale
        H = (h - self.h_off) / self.h_scale
        q = np.array([
            1, A, B, H, A * B,
            A * H, B * H, A ** 2, B ** 2, H ** 2,
            A * B * H, A ** 3, A * B ** 2, A * H ** 2, A ** 2 * B,
            B ** 3, B * H ** 2, A ** 2 * H, B ** 2 * H, H ** 3
        ])
        L = np.dot(self.l_num_c, q) / np.dot(self.l_den_c, q)
        P = np.dot(self.p_num_c, q) / np.dot(self.p_den_c, q)
        line = self.l_scale * L + self.l_off
        pixel = self.p_scale * P + self.p_off
        return int(pixel), int(line)
