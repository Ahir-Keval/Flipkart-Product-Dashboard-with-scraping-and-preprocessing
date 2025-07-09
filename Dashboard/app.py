from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sqlalchemy import create_engine
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)

#Connection 
engine = create_engine("mysql+pymysql://root:@localhost/csedept")
df = pd.read_sql("SELECT * FROM watches", engine)

df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce").fillna(0).astype(float)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
brands = df["Brand"].dropna().unique()

#to filter selected brands
def filter_by_brands(selected_brands):
    return df[df["Brand"].isin(selected_brands)]

#Routes
@app.route('/')
def index():
    return render_template('index.html', brands=brands)

@app.route('/overview', methods=['GET', 'POST'])
def overview():
    if request.method == 'POST':
        selected_brands = request.form.getlist('brands')

        if not selected_brands:
            return render_template(
                'overview.html',
                brands=brands,
                selected_brands=[],
                error_message="Please select at least one brand.",
                total_watches=None,
                avg_price=0,
                avg_discount=0,
                chart_url=None,
                price_dist_url=None
            )

        filtered_df = filter_by_brands(selected_brands)
        total_watches = len(filtered_df)
        avg_price = filtered_df['Price'].mean() if not filtered_df['Price'].isnull().all() else 0
        avg_discount = filtered_df['Discount'].mean() if not filtered_df['Discount'].isnull().all() else 0

        fig, ax = plt.subplots(figsize=(10, 6))
        brand_counts = filtered_df["Brand"].value_counts()
        sns.barplot(x=brand_counts.index, y=brand_counts.values, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close(fig)
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode()

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_df["Price"], bins=30, kde=True, ax=ax2)
        plt.tight_layout()
        img2 = BytesIO()
        plt.savefig(img2, format='png')
        plt.close(fig2)
        img2.seek(0)
        price_dist_url = base64.b64encode(img2.getvalue()).decode()

        return render_template(
            'overview.html',
            brands=brands,
            selected_brands=selected_brands,
            total_watches=total_watches,
            avg_price=avg_price,
            avg_discount=avg_discount,
            chart_url=chart_url,
            price_dist_url=price_dist_url,
            error_message=None
        )

    return render_template(
        'overview.html',
        brands=brands,
        selected_brands=[],
        total_watches=None,
        avg_price=0,
        avg_discount=0,
        chart_url=None,
        price_dist_url=None,
        error_message=None
    )


@app.route('/discount', methods=['GET', 'POST'])
def discount():
    if request.method == 'POST':
        selected_brands = request.form.getlist('brands')

        if not selected_brands:
            return render_template(
                'discount.html',
                brands=brands,
                error_message="Please select at least one brand.",
                top_discounts=None,
                chart_url=None,
                selected_brands=[]
            )

        filtered_df = df[df["Brand"].isin(selected_brands)]
        filtered_df = filtered_df.drop_duplicates(subset=['Name'])
        top_discounts = filtered_df.sort_values(by="Discount", ascending=False).head(10)

        if not filtered_df.empty:
            fig, ax = plt.subplots()
            sns.scatterplot(data=filtered_df, x="Price", y="Discount", ax=ax)
            plt.tight_layout()
            img = BytesIO()
            plt.savefig(img, format='png')
            plt.close(fig)
            img.seek(0)
            chart_url = base64.b64encode(img.getvalue()).decode()
        else:
            chart_url = None

        return render_template(
            'discount.html',
            brands=brands,
            top_discounts=top_discounts,
            chart_url=chart_url,
            selected_brands=selected_brands,
            error_message=None
        )

    return render_template(
        'discount.html',
        brands=brands,
        top_discounts=None,
        chart_url=None,
        selected_brands=[],
        error_message=None
    )
@app.route('/api/discounts_plot', methods=['GET'])
def api_discounts_plot():
    try:
        brands = request.args.get('brands')

        if not brands:
            return jsonify({'error': 'No brands selected'}), 400

        brands = brands.split(',')
        filtered_df = df[df['Brand'].isin(brands)]

        if filtered_df.empty:
            return jsonify({'error': 'No data found for the selected brands.'}), 400

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(filtered_df['Price'], filtered_df['Discount'], alpha=0.5, color='teal')
        ax.set_xlabel('Price (₹)')
        ax.set_ylabel('Discount (%)')
        ax.set_title('Price vs Discount')

        scatter_img = BytesIO()
        plt.tight_layout()
        plt.savefig(scatter_img, format='png')
        scatter_img.seek(0)
        scatter_img_base64 = base64.b64encode(scatter_img.getvalue()).decode('utf-8')
        plt.close(fig)

        avg_discount = filtered_df.groupby('Brand')['Discount'].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        avg_discount.plot(kind='bar', ax=ax, color='coral')
        ax.set_xlabel('Brand')
        ax.set_ylabel('Average Discount (%)')
        ax.set_title('Average Discount by Brand')

        bar_img = BytesIO()
        plt.savefig(bar_img, format='png')
        bar_img.seek(0)
        bar_img_base64 = base64.b64encode(bar_img.getvalue()).decode('utf-8')
        plt.close(fig)

        return jsonify({
            'scatter_chart_url': scatter_img_base64,
            'bar_chart_url': bar_img_base64
        })

    except Exception as e:
        return jsonify({'error': f"Error generating charts: {str(e)}"}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        rating = float(request.form['rating'])
        algorithm = request.form['algorithm']

        prediction = make_prediction(rating, algorithm)

        return render_template('predict.html', predicted_price=prediction)

    return render_template('predict.html')

def make_prediction(rating, algorithm):
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    prediction_df = df[["Rating", "Price"]].dropna()

    X = prediction_df[["Rating"]]
    y = prediction_df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm == 'Linear Regression':
        model = LinearRegression()
    elif algorithm == 'Decision Tree':
        model = DecisionTreeRegressor()
    elif algorithm == 'Random Forest':
        model = RandomForestRegressor()
    elif algorithm == 'KNN':
        model = KNeighborsRegressor()

    model.fit(X_train, y_train)
    predicted_price = model.predict([[rating]])[0]

    return round(predicted_price, 2)

@app.route('/api/overview', methods=['GET'])
def api_overview():
    brands_param = request.args.get('brands')
    if not brands_param:
        return jsonify({"error": "No brands selected"}), 400

    selected_brands = brands_param.split(',')
    filtered_df = filter_by_brands(selected_brands)

    if filtered_df.empty:
        return jsonify({
            "total_watches": 0,
            "avg_price": 0,
            "avg_discount": 0,
            "chart_url": None,
            "price_dist_url": None
        })

    total_watches = len(filtered_df)
    avg_price = filtered_df['Price'].mean()
    avg_discount = filtered_df['Discount'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    brand_counts = filtered_df["Brand"].value_counts()
    sns.barplot(x=brand_counts.index, y=brand_counts.values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_df["Price"], bins=30, kde=True, ax=ax2)
    plt.tight_layout()
    img2 = BytesIO()
    plt.savefig(img2, format='png')
    plt.close(fig2)
    img2.seek(0)
    price_dist_url = base64.b64encode(img2.getvalue()).decode()

    return jsonify({
        "total_watches": total_watches,
        "avg_price": avg_price,
        "avg_discount": avg_discount,
        "chart_url": chart_url,
        "price_dist_url": price_dist_url
    })
@app.route('/brand_insights', methods=['GET', 'POST'])
def brand_insights():
    if request.method == 'POST':
        selected_brands = request.json.get('brands', [])

        if not selected_brands:
            return jsonify({'error': 'Please select at least one brand.'})
        selected_brands = [brand.strip().lower() for brand in selected_brands]

        filtered_df = filter_by_brands(selected_brands)

        if filtered_df.empty:
            return jsonify({'error': 'No data found for the selected brands.'})

        top_discounts = filtered_df.sort_values(by="Discount", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=filtered_df, x="Price", y="Discount", hue="Brand", ax=ax)
        plt.tight_layout()

        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close(fig)
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode()

        return jsonify({
            'top_watches': top_discounts.to_dict(orient='records'),
            'scatter_chart_url': chart_url
        })

    return render_template(
        'brand_insights.html',
        brands=brands,
        selected_brands=[],
        top_discounts=None,
        chart_url=None,
        error_message=None
    )


@app.route('/api/brand_insights', methods=['POST'])
def get_brand_insights():
    data = request.get_json()
    selected_brands = data.get('brands')

    if not selected_brands:
        return jsonify({'error': 'No brands selected'}), 400

    filtered_df = df[df['Brand'].str.lower().isin([b.lower() for b in selected_brands])].copy()

    filtered_df['Name_clean'] = filtered_df['Name'].str.lower().str.strip()
    filtered_df = filtered_df.drop_duplicates(subset='Name_clean', keep='first')
    top_watches = filtered_df.sort_values(by='Discount', ascending=False).head(10)

    return jsonify({
        'top_watches': top_watches[['Name', 'Brand', 'Price', 'Discount']].to_dict(orient='records')
    })

@app.route('/api/brand_insights_plot', methods=['GET'])
def brand_insights_plot():
    try:
        brands = request.args.get('brands').split(',')

        filtered_df = df[df['Brand'].isin(brands)]

        if filtered_df.empty:
            return jsonify({'error': 'No data found for selected brands.'}), 400

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(filtered_df['Price'], filtered_df['Discount'], alpha=0.5, color='teal')
        ax.set_xlabel('Price (₹)')
        ax.set_ylabel('Discount (%)')
        ax.set_title('Price vs Discount')

        scatter_img = BytesIO()
        plt.tight_layout()
        plt.savefig(scatter_img, format='png')
        scatter_img.seek(0)
        scatter_img_base64 = base64.b64encode(scatter_img.getvalue()).decode('utf-8')
        plt.close(fig)

        avg_discount = filtered_df.groupby('Brand')['Discount'].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        avg_discount.plot(kind='bar', ax=ax, color='coral')
        ax.set_xlabel('Brand')
        ax.set_ylabel('Average Discount (%)')
        ax.set_title('Average Discount by Brand')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        bar_img = BytesIO()
        plt.savefig(bar_img, format='png')
        bar_img.seek(0)
        bar_img_base64 = base64.b64encode(bar_img.getvalue()).decode('utf-8')
        plt.close(fig)

        return jsonify({
            'scatter_chart_url': scatter_img_base64,
            'bar_chart_url': bar_img_base64
        })

    except Exception as e:
        return jsonify({'error': f"Error generating charts: {str(e)}"}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    rating = data.get("rating")
    algorithm = data.get("algorithm")

    if rating is None or algorithm not in ['Linear Regression', 'Decision Tree', 'Random Forest', 'KNN']:
        return jsonify({"error": "Invalid input"}), 400

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    prediction_df = df[["Rating", "Price"]].dropna()

    if prediction_df.empty:
        return jsonify({"error": "No valid data available for prediction"}), 400

    X = prediction_df[["Rating"]]
    y = prediction_df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        model = None
        if algorithm == 'Linear Regression':
            model = LinearRegression()
        elif algorithm == 'Decision Tree':
            model = DecisionTreeRegressor()
        elif algorithm == 'Random Forest':
            model = RandomForestRegressor()
        elif algorithm == 'KNN':
            model = KNeighborsRegressor()

        if model is None:
            return jsonify({"error": "Invalid algorithm selected"}), 400

        model.fit(X_train, y_train)
        predicted_price = model.predict([[rating]])[0]
        return jsonify({"rating": rating, "predicted_price": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/discounts', methods=['POST'])
def api_get_top_discounts():
    data = request.get_json()
    brands = data.get("brands")

    if not brands or not isinstance(brands, list) or len(brands) == 0:
        return jsonify({"error": "Brand list is required and cannot be empty"}), 400

    filtered_df = df[df["Brand"].str.lower().isin([b.lower() for b in brands])].copy()

    if filtered_df.empty:
        return jsonify({"error": "No data found for the selected brands"}), 404

    filtered_df = filtered_df.drop_duplicates(subset=["Name", "Brand", "Price"])

    filtered_df["Name_clean"] = filtered_df["Name"].str.lower().str.strip()
    filtered_df = filtered_df.drop_duplicates(subset=["Name_clean"])

    top_discounts = filtered_df.sort_values(by="Discount", ascending=False).head(10)
    result = top_discounts[["Brand", "Name", "Price", "Discount"]].to_dict(orient="records")

    return jsonify(result)
@app.route('/api/clustering', methods=['GET'])
def api_clustering():
    features = df[['Price', 'Discount', 'Rating']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    cluster_stats = df.groupby('Cluster')[['Price', 'Discount', 'Rating']].mean().reset_index()
    cluster_data = cluster_stats.to_dict(orient='records')
    return jsonify(cluster_data)


@app.route('/clustering')
def clustering():
    return render_template('clustering.html')

if __name__ == '__main__':
    app.run(debug=True)
   
