# written in python 3.7.3
import json
import random

import geojsonio
import numpy as np
import pandas as pd
from geojson import LineString, Point, Feature, FeatureCollection
from math import cos, asin, sqrt, degrees, atan2

import matplotlib.pyplot as plt


random_seed = 42
random.seed(random_seed)

p = 0.017453292519943295  # Pi / 180
r = 12742000  # Earth's radius is ~6371km => r = 2 * Earth's radius

class GeoPoint:
    def __init__(self, lng: float, lat: float, name_: str = None):
        # Why 5 digits? According to https://en.wikipedia.org/wiki/Decimal_degrees it's 1m. accuracy.
        self.lng = round(lng, 5)
        self.lat = round(lat, 5)
        self.name_ = name_
        
    def __repr__(self):
        # Copy-pastable format for most map applications
        name_str = f"{self.name_}, " if self.name_ is not None else ""
        return f"[{name_str}{self.lat}, {self.lng}]"
    
    def get_dist_from(self, other) -> int:
        # Return non-euclidean distance in meters, using Haversine formula
        # For more information on this formulation go to 
        a = (0.5 
             - cos((other.lat - self.lat) * p)/2 
             + (cos(self.lat * p) 
                * cos(other.lat * p) 
                * (1 - cos((other.lng - self.lng) * p)) / 2))
        d = int(r * asin(sqrt(a))) 
        return d
    
def get_all_geopoints_from_all_sets(all_sets):
    geo_points = [g
                for set_ in all_sets
                for g in set_]
    return geo_points

def get_random_geo_point(center_lat=32.1, center_lng=34.8, radius=0.1, name_=None):
    geo_point =  GeoPoint(lat = center_lat + (radius * random.random()), 
                        lng = center_lng + (radius * random.random()),
                        name_ = name_)
    return geo_point
    
def get_distances_array_and_set_to_points_dict(all_sets):
    all_geo_points = []
    set_to_points_dict = {}
    first_point_idx = 0
    for idx, set_ in enumerate(all_sets):
        all_geo_points += set_
        n_points_in_set = len(set_)
        set_to_points_dict[idx] = list(range(first_point_idx, first_point_idx + n_points_in_set))
        first_point_idx += n_points_in_set

    n_points = first_point_idx
    distances_array = np.array([[all_geo_points[i].get_dist_from(all_geo_points[j])
                                for i in range(n_points)]
                                for j in range(n_points)])

    return all_geo_points, set_to_points_dict, distances_array

def get_random_geo_point(center_lat=32.1, center_lng=34.8, radius=0.1, name_=None):
    geo_point =  GeoPoint(lat = center_lat + (radius * random.random()), 
                        lng = center_lng + (radius * random.random()),
                        name_ = name_)
    return geo_point

def generate_random_input_in_geo_points(n_sets: int, poisson_lambda: int = 2) -> [{int: int}, {int: int}, np.array]:
    set_to_points_dict = {}
    first_point_idx = 0
    for set_idx in range(n_sets):
        n_points_in_set = 1 + np.random.poisson(poisson_lambda)
        set_to_points_dict[set_idx] = list(range(first_point_idx, first_point_idx + n_points_in_set))
        first_point_idx += n_points_in_set

    n_points = first_point_idx
    all_sets = []
    for idx_set in range(n_sets):
        all_sets += [[get_random_geo_point(name_=f's{idx_set}_i{idx_point}') 
                    for idx_point in range(len(set_to_points_dict[idx_set]))]]
    
    all_geo_points = get_all_geopoints_from_all_sets(all_sets)
    distances_array = np.array([[all_geo_points[i].get_dist_from(all_geo_points[j])
                                for i in range(n_points)]
                                for j in range(n_points)])

    return all_sets, all_geo_points, set_to_points_dict, distances_array

def get_features_for_all_points(all_sets):
    points = []
    for set_ in all_sets:
        color = "#" + ''.join(random.choices('0123456789abcdef', k=6))
        points += [
            Feature(geometry=Point(tuple([g.lng, g.lat])),
                    properties={"name": g.name_,
                                "marker-symbol": int(g.name_[-1]),
                                "marker-color": color})
            for g in set_]
    return points


def plot_route_on_map(all_sets, optimal_path_in_points_idxs):
    points = get_features_for_all_points(all_sets)
    
    all_geo_points = get_all_geopoints_from_all_sets(all_sets)
    lng_lat_list = [tuple([all_geo_points[i].lng, all_geo_points[i].lat])
                    for i in optimal_path_in_points_idxs]
    route = Feature(geometry=LineString(lng_lat_list),
                    properties={"name": "This is our route",
                                "stroke": "black"})
    
    feature_collection = FeatureCollection(features=points+[route])
    geojsonio.display(json.dumps(feature_collection));
    
def retrace_optimal_path(memo: dict, n_sets: int) -> [[int], [int], float]:
    sets_to_retrace = tuple(range(n_sets))

    full_path_memo = dict((k, v) for k, v in memo.items() if k[0] == sets_to_retrace)
    path_key = min(full_path_memo.keys(), key=lambda x: full_path_memo[x][0])

    _, last_set, last_point = path_key
    optimal_cost, next_to_last_set, next_to_last_point = memo[path_key]

    optimal_path_in_points_idxs = [last_point]
    optimal_path_in_sets_idxs = [last_set]
    sets_to_retrace = tuple(sorted(set(sets_to_retrace).difference({last_set})))

    while next_to_last_set is not None:
        last_point = next_to_last_point
        last_set = next_to_last_set
        path_key = (sets_to_retrace, last_set, last_point)
        _, next_to_last_set, next_to_last_point = memo[path_key]

        optimal_path_in_points_idxs = [last_point] + optimal_path_in_points_idxs
        optimal_path_in_sets_idxs = [last_set] + optimal_path_in_sets_idxs
        sets_to_retrace = tuple(sorted(set(sets_to_retrace).difference({last_set})))

    return optimal_path_in_points_idxs, optimal_path_in_sets_idxs, optimal_cost
    
    
def DP_Set_TSP(set_to_points_dict, distances_array):
    all_sets = set(set_to_points_dict.keys())
    n_sets = len(all_sets)

    # memo keys: tuple(sorted_sets_in_path, last_set_in_path, last_point_in_path)
    # memo values: tuple(cost_thus_far, next_to_last_set_in_path, next_to_last_point_in_path)
    memo = {(tuple([set_idx]), set_idx, p_idx): tuple([0, None, None])
            for set_idx, points_idxs in set_to_points_dict.items()
            for p_idx in points_idxs}
    queue = [(tuple([set_idx]), set_idx, p_idx)
             for set_idx, points_idxs in set_to_points_dict.items()
             for p_idx in points_idxs]

    while queue:
        prev_visited_sets, prev_last_set, prev_last_point = queue.pop(0)
        prev_dist, _, _ = memo[(prev_visited_sets, prev_last_set, prev_last_point)]

        to_visit = all_sets.difference(set(prev_visited_sets))
        for new_last_set in to_visit:
            new_visited_sets = tuple(sorted(list(prev_visited_sets) + [new_last_set]))
            for new_last_point in set_to_points_dict[new_last_set]:
                new_dist = prev_dist + distances_array[prev_last_point][new_last_point]

                new_key = (new_visited_sets, new_last_set, new_last_point)
                new_value = (new_dist, prev_last_set, prev_last_point)

                if new_key not in memo:
                    memo[new_key] = new_value
                    queue += [new_key]
                else:
                    if new_dist < memo[new_key][0]:
                        memo[new_key] = new_value

    optimal_path_in_points_idxs, optimal_path_in_sets_idxs, optimal_cost = retrace_optimal_path(memo, n_sets)

    return optimal_path_in_points_idxs, optimal_path_in_sets_idxs, optimal_cost

n_sets = 3
poisson_lambda = 1
all_sets, all_geo_points, set_to_points_dict, distances_array = generate_random_input_in_geo_points(n_sets, poisson_lambda)
optimal_path_in_points, optimal_path_in_sets, optimal_cost = DP_Set_TSP(set_to_points_dict, distances_array)
print(optimal_path_in_points, optimal_path_in_sets, optimal_cost)


plot_route_on_map(all_sets, optimal_path_in_points)
