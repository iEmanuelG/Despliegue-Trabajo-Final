import streamlit as st
import pandas as pd
import joblib
import os

# Set a wider layout
st.set_page_config(layout="wide")

st.title("Predicción de estado de desaparición")

# Function to load the model and encoder
@st.cache_resource
def load_resources():
    try:
        model_filename = 'model.pkl'
        encoder_filename = 'onehot_encoder.pkl'

        # Check if files exist in the current directory
        if not os.path.exists(model_filename):
            st.error(f"Error: Model file '{model_filename}' not found.")
            return None, None
        if not os.path.exists(encoder_filename):
             st.error(f"Error: Encoder file '{encoder_filename}' not found.")
             return None, None

        loaded_model = joblib.load(model_filename)
        loaded_encoder = joblib.load(encoder_filename)
        return loaded_model, loaded_encoder
    except Exception as e:
        st.error(f"Error loading model or encoder: {e}")
        return None, None

loaded_model, loaded_encoder = load_resources()

if loaded_model is not None and loaded_encoder is not None:
    st.write("Model and encoder loaded successfully.")

    st.subheader("Cargar archivo CSV")
    uploaded_file = st.file_uploader("Sube tu archivo CSV (asegúrate de que contenga las columnas necesarias)", type="csv")

    if uploaded_file is not None:
        try:
            data_ini = pd.read_csv(uploaded_file, delimiter=',')
            st.write("Datos cargados:")
            st.dataframe(data_ini.head())

            # Store original data for later use (displaying results)
            original_data_for_display = data_ini.copy()

            # ## Limpieza y preparación de datos
            # Applying the same data cleaning steps as in the notebook

            # Filter out 'Desaparecido' status
            data_ini = data_ini[data_ini["Estado de la desaparición"] != "Desaparecido"]

            # Filter for Colombia
            data_ini = data_ini[data_ini["País donde ocurre la desaparición"] == "Colombia"]

            # Define columns to drop
            columns_to_drop = ["ID",
                "Entidad que realiza el registro de la desaparición",
                "Pueblo indígena del desaparecido",
                "País de nacimiento del desaparecido",
                "Fecha de la desaparición",
                "Año de la desaparición",
                "Mes de la desaparición",
                "Día de la desaparición",
                "Codigo Dane Departamento",
                "Codigo Dane Municipio",
                "Municipio donde ocurre la desaparición DANE",
                "Localidad donde ocurre la desaparición",
                "Contexto",
                "Grupo de edad quinquenal del desaparecido",
                "Grupo mayor y menor de edad del desaparecido",
                "Grupo de edad judicial del desaparecido",
                "País donde ocurre la desaparición",
                "Transgénero",
                "Pertenencia étnica del desaparecido",
                "Orientación sexual del desaparecido",
                "Identidad de género del desaparecido",
                "Escolaridad del desaparecido",
                "Estado de la desaparición" # Drop target variable before encoding
            ]

            # Drop the columns
            data = data_ini.drop(columns=columns_to_drop, errors='ignore')

            # Identify categorical columns
            categorical_cols_initial = data.select_dtypes(include=['object']).columns.tolist()

            # Convert object columns to category
            for col in categorical_cols_initial:
                 data[col] = data[col].astype('category')

            categorical_cols_after_conversion = data.select_dtypes(include=['category']).columns.tolist()


            # Ensure the columns match the encoder's expected features and apply the same drop_first logic
            data_for_encoding = data[categorical_cols_after_conversion]

            # Apply the loaded encoder
            try:
                encoded_features = loaded_encoder.transform(data_for_encoding)
                encoded_df = pd.DataFrame(encoded_features, columns=loaded_encoder.get_feature_names_out(categorical_cols_after_conversion))

                # Apply drop_first for the specific columns 'Sexo del desaparecido' and 'Clasificación de la desaparición'
                # Identify the columns that correspond to 'Sexo del desaparecido' and 'Clasificación de la desaparición'
                sexo_cols_to_drop = [col for col in encoded_df.columns if 'Sexo del desaparecido_' in col]
                clasificacion_cols_to_drop = [col for col in encoded_df.columns if 'Clasificación de la desaparición_' in col]

                # Drop the same column that was dropped during training (assuming the first one alphabetically)
                if len(sexo_cols_to_drop) > 1:
                    encoded_df = encoded_df.drop(sorted(sexo_cols_to_drop)[0], axis=1)
                if len(clasificacion_cols_to_drop) > 1:
                    encoded_df = encoded_df.drop(sorted(clasificacion_cols_to_drop)[0], axis=1)

                # Separate numeric features
                features_numeric = data.select_dtypes(exclude=['category', 'object'])

                # Concatenate the numeric features with the dummy variables
                data_processed = pd.concat([features_numeric.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

                st.write("Datos procesados para la predicción:")
                st.dataframe(data_processed.head())
                st.write(f"Shape of processed data: {data_processed.shape}")


                # Ensure columns in processed data match the training data columns of the model
                # (This is a crucial step if column order or presence differs)
                # One way to handle this is to save the training columns and reindex here.
                # For simplicity, let's assume the columns match for now, but for production,
                # you'd need a robust way to handle potential column mismatches.
                # e.g., data_processed = data_processed.reindex(columns=training_columns, fill_value=0)

                # Make predictions
                predictions = loaded_model.predict(data_processed)

                # Add predictions to the original dataframe for display
                original_data_for_display = original_data_for_display.iloc[data_ini.index] # Align with filtered data
                original_data_for_display['Predicción'] = predictions

                # Map numerical predictions back to category names
                prediction_mapping = {1: 'Aparece vivo', 0: 'Aparece muerto'}
                original_data_for_display['Predicción'] = original_data_for_display['Predicción'].map(prediction_mapping)

                # Select and display the relevant columns
                result_table = original_data_for_display[['Estado de la desaparición', 'Predicción']]

                st.subheader("Resultados de la Predicción")
                st.dataframe(result_table)

            except ValueError as ve:
                st.error(f"Error during transformation: {ve}. This might indicate a mismatch between the columns in your uploaded data and the data used to train the encoder.")
                st.write("Please ensure your uploaded CSV has the expected columns and categories.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

        except Exception as e:
            st.error(f"Error reading or processing the CSV file: {e}")
            st.write("Please ensure the file is a valid CSV with the correct delimiter (',') and columns.")
