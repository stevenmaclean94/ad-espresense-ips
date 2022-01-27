"""

Credits to:

https://github.com/glucee/Multilateration/blob/master/Python/example.py

Uses:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html#optimize-minimize-neldermead

"""

import mqttapi as mqtt
from scipy.optimize import minimize
import numpy as np
import json
from pykalman import KalmanFilter
class ESPresenseIps(mqtt.Mqtt):
    def initialize(self):
        self.set_namespace("mqtt")
        self.devices = {}
        for device in self.args["devices"]:
          self.devices.setdefault(device["id"],{})["name"]=device["name"]

        self.initial_state = True
        self.x = None
        self.P = None
        for room, pos in self.args["rooms"].items():
            t = f"{self.args.get('rooms_topic', 'espresense/rooms')}/{room}"
            self.log(f"Subscribing to topic {t}")
            self.mqtt_unsubscribe(t)
            self.mqtt_subscribe(t)
            self.listen_event(self.mqtt_message, "MQTT_MESSAGE", topic=t)

    def mqtt_message(self, event_name, data, *args, **kwargs):
        """Process a message sent on the MQTT Topic."""
        topic = data.get("topic")
        payload = data.get("payload")

        topic_path = topic.split("/")
        room = topic_path[-1].lower()

        payload_json = {}
        try:
            payload_json = json.loads(payload)
        except ValueError:
            pass

        id = payload_json.get("id")
        name = payload_json.get("name")
        distance = payload_json.get("distance")
        self.log(f"{id} {room} {distance}", level="DEBUG")

        device = self.devices.setdefault(id,{})
        device["measures"] = device.get("measures", 0) + 1

        if (room in self.args["rooms"]):
            dr = device.setdefault("rooms",{}).setdefault(room,{"pos":self.args["rooms"][room]})
            dr["distance"] = distance

            distance_to_stations=[]
            stations_coordinates=[]
            for r in device["rooms"]:
                if "distance" in device["rooms"][r]:
                    distance_to_stations.append(device["rooms"][r]["distance"])
                    stations_coordinates.append(device["rooms"][r]["pos"])

            name = device.get("name", name)
            if (name) and len(distance_to_stations)>2:
                device["x0"] = position_solve(distance_to_stations, np.array(stations_coordinates), device.get("x0", None))
                pos = device["x0"].tolist()
                if self.initial_state:
                    self.kf = self.initialize_kalman(np.asarray(pos)) # can't do in initialize() because we need the inital state estimate
                    self.initial_state = False
                else:
                    self.filter(pos[0:2])
                    pos[0] = self.x[0]
                    pos[1] = self.x[2]
                roomname, dist = room_solve(self,round(pos[0],2,),round(pos[1],2))
                self.mqtt_publish(f"{self.args.get('room_topic', 'espresense/ips/rooms')}/{roomname}", json.dumps({"id":id, "distance": dist, "x":round(pos[0],2),"y":round(pos[1],2),"z":round(pos[2],2), "fixes":len(distance_to_stations),"measures":device["measures"]}))
                self.mqtt_publish(f"{self.args.get('ips_topic', 'espresense/ips')}/{id}", json.dumps({"name":name, "x":round(pos[0],2),"y":round(pos[1],2),"z":round(pos[2],2), "fixes":len(distance_to_stations),"measures":device["measures"],"currentroom":roomname}))
                self.mqtt_publish(f"{self.args.get('location_topic', 'espresense/location')}/{id}", json.dumps({"name":name, "longitude":(self.config["longitude"]+(pos[0]/111111)),"latitude":(self.config["latitude"]+(pos[1]/111111)),"elevation":(self.config.get("elevation","0")+pos[2]), "fixes":len(distance_to_stations),"measures":device["measures"]}))
    
    def filter(self, obs):
        (self.x, self.P) = self.kf.filter_update(filtered_state_mean = self.x, filtered_state_covariance = self.P, observation=obs)
        return self.x

    def initialize_kalman(self, pos):
        """Super basic model of just velocity -> position filter. The covariance were estimated offline with the pykalman em algo"""
        trans_matrix = np.array([[1, 1, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 1],
                                 [0, 0, 0, 1]])
        trans_cov = np.array([[3.08552115e-01, 8.87165238e-02, 1.38591598e-02, 1.59152558e-04],
                              [8.87165238e-02, 2.20473917e-01, 7.04238191e-04, 5.18411870e-03],
                              [1.38591598e-02, 7.04238191e-04, 2.99125145e-01, 8.93007865e-02],
                              [1.59152558e-04, 5.18411870e-03, 8.93007865e-02, 2.19261020e-01]])
        trans_offsets = np.array([0,0,0,0])

        obs_matrix = np.array([[1, 0, 0, 0],
                                [0, 0, 1, 0]])
        obs_cov = np.array([[0.14574861, 0.00815694],
                            [0.00815694, 0.13348215]])
        obs_offsets = np.array([0, 0])

        init_state_mean = np.array([pos[0],0,pos[1],0])
        init_state_cov = np.array([[ 0.05392465, -0.02642348,  0.00137511, -0.00054785],
                                   [-0.02642348,  0.09629353, -0.00052511,  0.00099813],
                                   [ 0.00137511, -0.00052511,  0.05205013, -0.0256734 ],
                                   [-0.00054785,  0.00099813, -0.0256734 ,  0.09548006]])
        self.x = init_state_mean
        self.P = init_state_cov
        return KalmanFilter(transition_matrices = trans_matrix,
                            observation_matrices = obs_matrix,
                            initial_state_mean = init_state_mean,
                            initial_state_covariance = init_state_cov,
                            observation_covariance = obs_cov,
                            transition_covariance = trans_cov,
                            transition_offsets = trans_offsets,
                            observation_offsets = obs_offsets)


def position_solve(distances_to_station, stations_coordinates, last):
    def error(x, c, r):
        return sum([(np.linalg.norm(x - c[i]) - r[i]) ** 2 for i in range(len(c))])

    l = len(stations_coordinates)
    S = sum(distances_to_station)
    # compute weight vector for initial guess
    W = [((l - 1) * S) / (S - w) for w in distances_to_station]
    # get initial guess of point location
    x0 = last if last is not None else sum([W[i] * stations_coordinates[i] for i in range(l)])
    # optimize distance from signal origin to border of spheres
    return minimize(
        error,
        x0,
        args=(stations_coordinates, distances_to_station),
        method="Nelder-Mead",
        options={'xatol': 0.001, 'fatol': 0.001, 'adaptive': True}
    ).x

def room_solve(self, xpos, ypos):
    for rooms in self.args["roomplans"]:
        if rooms["x1"] < float(xpos) < rooms["x2"] and rooms["y1"] < float(ypos) < rooms["y2"]:
            dist = np.sqrt((float(xpos) - rooms["x1"])**2 +(float(ypos) - rooms["y1"])**2)
            return rooms["name"], dist
    return "none"

