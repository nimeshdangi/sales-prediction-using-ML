from flask import Flask, request, render_template, flash, redirect, url_for, session
import pickle
import pandas as pd
from xgboost import XGBRegressor

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'my_secret_key'

# # Load the models
xgb_loaded_model = XGBRegressor()
xgb_loaded_model.load_model('./models/xgb_tuned_model.json')

with open('./models/lasso_pipeline.pkl', 'rb') as f:
    LR_Lasso = pickle.load(f)

with open('./models/rand_forest_tuned_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)


# Route for home page (optional)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        session['form_data'] = request.form.to_dict()
        # Get form data
        item_weight = request.form['item_weight']
        try:
            item_weight = float(item_weight)
            if item_weight > 0:
                item_weight = item_weight
            else:
                flash("Error: Item weight cannot be negative or zero!")
                return redirect(url_for('home'))

        except ValueError:
            flash(f"Please enter a valid item weight. You entered '{item_weight}'.")
            return redirect(url_for('home'))
        item_fat_content = request.form['item_fat_content']
        item_visibility = request.form['item_visibility']
        try:
            item_visibility = float(item_visibility)
            if 0 <= item_visibility <= 0.2:
                item_visibility = item_visibility
            else:
                flash("Error: Item Visibility must be between 0 and 0.2.")
                return redirect(url_for('home'))

        except ValueError:
            flash(f"Please enter a valid number (0-0.2) for Item Visibility. You entered '{item_visibility}'.", "error")
            return redirect(url_for('home'))
        item_mrp = request.form['item_mrp']
        try:
            item_mrp = float(item_mrp)
            if item_mrp > 0:
                item_mrp = item_mrp
            else:
                flash("Error: Item MRP cannot be a negative number.")
                return redirect(url_for('home'))
        except ValueError:
            flash(f"Error: Please enter a valid MRP. You entered '{item_mrp}'.", "error")
            return redirect(url_for('home'))
        item_type = request.form['item_type']
        outlet_establishment_year = request.form['outlet_establishment_year']
        try:
            outlet_establishment_year = int(outlet_establishment_year)
            if outlet_establishment_year < 0 or 2024 - outlet_establishment_year < 0 or outlet_establishment_year < 1985:
                flash(f"Error: Please enter a valid year! You entered: {outlet_establishment_year}")
                return redirect(url_for('home'))
            else:
                outlet_establishment_year = outlet_establishment_year
        except ValueError:
            flash(f"Error: Please enter a valid year! You entered {outlet_establishment_year}")
            return redirect(url_for('home'))

        outlet_size = request.form['outlet_size']
        outlet_location_type = request.form['outlet_location_type']
        outlet_type = request.form['outlet_type']

        # Create a DataFrame from the form data
        input_data = pd.DataFrame({
            'Item_Weight': [item_weight],
            'Item_Fat_Content': [item_fat_content],
            'Item_Visibility': [item_visibility],
            'Item_Identifier_Categories': [item_type],
            'Item_MRP': [item_mrp],
            'Outlet_Establishment_Year': [outlet_establishment_year],
            'Outlet_Size': [outlet_size],
            'Outlet_Location_Type': [outlet_location_type],
            'Outlet_Type': [outlet_type]
        })

        # Preprocess the input data (e.g., encoding categorical features, scaling numerical features)
        # This is an example. Adjust this according to how your model was trained.
        input_data = preprocess_input_data(input_data)

        # Make a prediction
        predicted_sales = []
        linear_regression_model_predicted_sales = LR_Lasso.predict(input_data)
        predicted_sales.append(linear_regression_model_predicted_sales)
        random_forest_model_predicted_sales = random_forest_model.predict(input_data)
        predicted_sales.append(random_forest_model_predicted_sales)
        xg_boost_model_predicted_sales = xgb_loaded_model.predict(input_data)
        predicted_sales.append(xg_boost_model_predicted_sales)

        form_data = session.pop('form_data', {})
        return render_template('index.html', linear_prediction=round(predicted_sales[0][0], 2),
                               rf_prediction=round(predicted_sales[1][0], 2),
                               xg_prediction=round(predicted_sales[2][0], 2), prediction=predicted_sales, form_data=form_data)
    form_data = session.pop('form_data', {})
    return render_template('index.html', prediction=None, form_data=form_data)


def preprocess_input_data(data):
    # getting the amount of established years in new column and deleting old column
    data['Outlet_Age'] = 2024 - data['Outlet_Establishment_Year']
    del data['Outlet_Establishment_Year']
    data['Outlet_Size'] = data['Outlet_Size'].map({'Small': 1,
                                                   'Medium': 2,
                                                   'High': 3
                                                   }).astype(int)
    data['Outlet_Location_Type'] = data['Outlet_Location_Type'].str[-1:].astype(int)
    item_fat_content_mapping = {
        'Low Fat': 0,
        'Regular': 1,
    }
    outlet_type_mapping = {
        'Grocery Store': 0,
        'Supermarket Type1': 1,
        'Supermarket Type2': 2,
        'Supermarket Type3': 3
    }
    outlet_location_type_mapping = {
        1: 0,
        2: 1,
        3: 2
    }

    # Apply the mapping to the column
    data['Item_Fat_Content'] = data['Item_Fat_Content'].map(item_fat_content_mapping)
    data['Outlet_Type'] = data['Outlet_Type'].map(outlet_type_mapping)
    data['Outlet_Location_Type'] = data['Outlet_Location_Type'].map(outlet_location_type_mapping)

    with open('./training-columns/training_columns.pkl', 'rb') as col_file:
        training_columns = pickle.load(col_file)

    # One-hot encode the inference data
    inference_encoded = pd.get_dummies(data, columns=['Item_Identifier_Categories'], drop_first=False)
    # Align columns with training data
    inference_encoded = inference_encoded.reindex(columns=training_columns, fill_value=False)

    return inference_encoded


if __name__ == "__main__":
    app.run(debug=True)
