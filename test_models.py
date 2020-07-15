from pickle import load

voltage_model = load(open("voltage model filename", "rb"))
current_model = load(open("current model filename", "rb"))

# get the x input from inline arguments as well
# standardize the input before the models are used on it (??)

# voltage_prediction = voltage_model.predict(x_standardized)
# current_prediction = current_model.predict(x_standardized)

# from pickle import dump
# save the model
# dump(model, open('model.pkl', 'wb'))
# save the scaler
# dump(scaler, open('scaler.pkl', 'wb'))

# from pickle import load
# load the model
# model = load(open('model.pkl', 'rb'))
# load the scaler
# scaler = load(open('scaler.pkl', 'rb'))
