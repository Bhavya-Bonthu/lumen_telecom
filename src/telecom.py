import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import hashlib
from urllib.parse import quote_plus
from pymongo import MongoClient
from bson.objectid import ObjectId
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import io
