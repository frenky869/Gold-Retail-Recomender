import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import math

# Page configuration
st.set_page_config(
    page_title="Gold Retail Recommender - Nyatike Mine",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .metric-card {
        background: rgba(255, 215, 0, 0.1);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #FFD700;
        margin: 10px 0;
    }
    .recommendation-card {
        background: rgba(255, 215, 0, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 215, 0, 0.3);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class GoldRetailRecommender:
    def __init__(self):
        self.nyatike_coords = (-0.8996, 34.2986)  # Nyatike Gold Mine coordinates
        self.retailers_data = self.load_sample_data()
        
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates using Haversine formula"""
        R = 6371  # Earth radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
        
    def load_sample_data(self):
        """Load sample retailer data"""
        data = {
            'retailer_id': range(1, 21),
            'name': [
                'Kenya Gold Refinery', 'Nairobi Gold Exchange', 'Mombasa Precious Metals',
                'Kisumu Gold Buyers', 'Eldoret Gold Traders', 'Nakuru Gold Hub',
                'Thika Gold Merchants', 'Kitale Gold Exchange', 'Malindi Gold Center',
                'Meru Gold Traders', 'Nyeri Gold Buyers', 'Garissa Gold Market',
                'Kakamega Gold Exchange', 'Bungoma Gold Center', 'Busia Gold Traders',
                'Homabay Gold Market', 'Migori Gold Buyers', 'Siaya Gold Exchange',
                'Vihiga Gold Center', 'Transmara Gold Traders'
            ],
            'city': [
                'Nairobi', 'Nairobi', 'Mombasa', 'Kisumu', 'Eldoret', 'Nakuru',
                'Thika', 'Kitale', 'Malindi', 'Meru', 'Nyeri', 'Garissa',
                'Kakamega', 'Bungoma', 'Busia', 'Homabay', 'Migori', 'Siaya',
                'Vihiga', 'Narok'
            ],
            'price_per_gram': [7200, 7100, 6900, 6800, 7000, 6950, 7050, 6850, 6750, 6950,
                             7100, 6650, 6800, 6700, 6750, 6850, 6900, 6700, 6800, 6950],
            'purchase_capacity_kg': [45, 35, 25, 20, 30, 28, 32, 22, 18, 26, 
                                   38, 15, 24, 19, 21, 23, 27, 17, 20, 29],
            'trust_rating': [4.8, 4.7, 4.5, 4.3, 4.6, 4.4, 4.5, 4.2, 4.1, 4.3,
                           4.7, 4.0, 4.3, 4.1, 4.2, 4.4, 4.5, 4.0, 4.2, 4.6],
            'years_in_business': [25, 20, 15, 12, 18, 14, 16, 10, 8, 13, 
                                22, 6, 11, 9, 10, 13, 15, 7, 9, 17],
            'certification': ['LBMA', 'ISO', 'Local', 'LBMA', 'ISO', 'Local', 'LBMA', 
                            'ISO', 'Local', 'LBMA', 'ISO', 'Local', 'LBMA', 'ISO', 
                            'Local', 'LBMA', 'ISO', 'Local', 'LBMA', 'ISO'],
            'latitude': [
                -1.2921, -1.2833, -4.0435, -0.1022, 0.5143, -0.3031,
                -1.0333, 1.0167, -3.2176, 0.0557, -0.4201, -0.4532,
                0.2827, 0.5695, 0.4602, -0.5367, -1.0634, 0.0607,
                0.0416, -1.0876
            ],
            'longitude': [
                36.8219, 36.8167, 39.6682, 34.7617, 35.2698, 36.0800,
                37.0833, 35.0000, 40.1167, 37.6452, 36.9476, 39.6461,
                34.7519, 34.5584, 34.1117, 34.4531, 34.4731, 34.2881,
                34.7250, 35.0933
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Calculate distances from Nyatike using our custom function
        df['distance_km'] = df.apply(
            lambda row: self.calculate_distance(
                self.nyatike_coords[0], self.nyatike_coords[1],
                row['latitude'], row['longitude']
            ), axis=1
        )
        
        return df
    
    def calculate_similarity_score(self, user_preferences, gold_quantity):
        """Calculate similarity scores based on user preferences"""
        features = ['price_per_gram', 'trust_rating', 'distance_km', 'purchase_capacity_kg']
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(self.retailers_data[features])
        
        # Calculate weights based on user preferences
        weights = np.array([
            user_preferences['price_importance'],
            user_preferences['trust_importance'],
            user_preferences['distance_importance'],
            user_preferences['capacity_importance']
        ])
        
        # Apply weights to normalized features
        weighted_features = normalized_features * weights
        
        # Calculate similarity scores
        scores = []
        for i in range(len(weighted_features)):
            score = np.sum(weighted_features[i])
            
            # Bonus for capacity if retailer can handle the quantity
            if self.retailers_data.iloc[i]['purchase_capacity_kg'] >= gold_quantity:
                score += 2
                
            # Bonus for certification
            if self.retailers_data.iloc[i]['certification'] in ['LBMA', 'ISO']:
                score += 1
                
            scores.append(score)
        
        self.retailers_data['similarity_score'] = scores
        return self.retailers_data
    
    def get_recommendations(self, user_preferences, gold_quantity, top_n=5):
        """Get top N recommendations"""
        scored_data = self.calculate_similarity_score(user_preferences, gold_quantity)
        
        # Sort by similarity score
        recommendations = scored_data.sort_values('similarity_score', ascending=False).head(top_n)
        
        return recommendations

def main():
    st.markdown('<div class="main-header">💰 Gold Retail Recommender</div>', unsafe_allow_html=True)
    st.markdown('### Find the Best Buyers for Your Nyatike Gold')
    
    # Initialize recommender
    recommender = GoldRetailRecommender()
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("🔧 Recommendation Settings")
        
        st.subheader("Gold Details")
        gold_quantity = st.slider("Gold Quantity (kg)", 0.1, 50.0, 5.0, 0.1)
        gold_purity = st.selectbox("Gold Purity", ["24K", "22K", "18K", "14K", "10K"])
        
        st.subheader("Your Preferences")
        price_importance = st.slider("Price Importance", 1, 10, 8)
        trust_importance = st.slider("Trust/Security Importance", 1, 10, 9)
        distance_importance = st.slider("Distance Importance", 1, 10, 5)
        capacity_importance = st.slider("Capacity Importance", 1, 10, 7)
        
        user_preferences = {
            'price_importance': price_importance,
            'trust_importance': trust_importance,
            'distance_importance': distance_importance,
            'capacity_importance': capacity_importance
        }
        
        # Additional filters
        st.subheader("Additional Filters")
        min_trust_rating = st.slider("Minimum Trust Rating", 3.0, 5.0, 4.0, 0.1)
        max_distance = st.slider("Maximum Distance (km)", 50, 500, 200)
        certification_pref = st.multiselect(
            "Preferred Certifications",
            ["LBMA", "ISO", "Local"],
            default=["LBMA", "ISO"]
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🗺️ Retailer Locations")
        
        # Create map
        try:
            fig = px.scatter_mapbox(
                recommender.retailers_data,
                lat="latitude",
                lon="longitude",
                hover_name="name",
                hover_data={
                    "price_per_gram": True,
                    "trust_rating": True,
                    "distance_km": True,
                    "city": True
                },
                color="trust_rating",
                size="purchase_capacity_kg",
                color_continuous_scale="viridis",
                zoom=6,
                height=500
            )
            
            # Add Nyatike mine location
            mine_data = pd.DataFrame({
                'latitude': [recommender.nyatike_coords[0]],
                'longitude': [recommender.nyatike_coords[1]],
                'name': ['Nyatike Gold Mine'],
                'trust_rating': [5.0],
                'purchase_capacity_kg': [10]
            })
            
            fig.add_trace(px.scatter_mapbox(
                mine_data,
                lat="latitude",
                lon="longitude",
                hover_name="name"
            ).data[0])
            
            fig.update_traces(
                marker=dict(size=20, symbol="diamond", color="red"),
                selector=dict(name="Nyatike Gold Mine")
            )
            
            fig.update_layout(
                mapbox_style="open-street-map",
                margin={"r":0,"t":0,"l":0,"b":0}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Map not available: {str(e)}")
            st.info("Displaying retailer list instead:")
            st.dataframe(recommender.retailers_data[['name', 'city', 'price_per_gram', 'distance_km']])
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        
        # Calculate some statistics
        avg_price = recommender.retailers_data['price_per_gram'].mean()
        max_price = recommender.retailers_data['price_per_gram'].max()
        min_distance = recommender.retailers_data['distance_km'].min()
        
        st.metric("Average Price (KSH/g)", f"{avg_price:,.0f}")
        st.metric("Highest Price (KSH/g)", f"{max_price:,.0f}")
        st.metric("Nearest Retailer (km)", f"{min_distance:,.0f}")
        st.metric("Total Retailers", len(recommender.retailers_data))
        
        st.markdown("---")
        st.markdown("### 💎 Gold Price Alert")
        
        # Simulated current gold price
        current_gold_price = 6500  # KSH per gram
        st.metric("Current Gold Price", f"KSH {current_gold_price:,.0f}/g")
        
        if st.button("🔄 Update Prices"):
            st.success("Prices updated successfully!")
    
    # Generate recommendations
    if st.button("🎯 Get Recommendations", type="primary"):
        with st.spinner("Finding the best gold retailers..."):
            try:
                recommendations = recommender.get_recommendations(user_preferences, gold_quantity)
                
                st.markdown("---")
                st.markdown("### 🏆 Top Recommendations")
                
                for i, (idx, retailer) in enumerate(recommendations.iterrows(), 1):
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**#{i} {retailer['name']}**")
                            st.markdown(f"📍 {retailer['city']} • 🛡️ {retailer['certification']} • ⭐ {retailer['trust_rating']:.1f}")
                            
                        with col2:
                            st.markdown(f"**💰 KSH {retailer['price_per_gram']:,.0f}/g**")
                            st.markdown(f"**📦 {retailer['purchase_capacity_kg']:.1f} kg cap**")
                            
                        with col3:
                            st.markdown(f"**🚗 {retailer['distance_km']:.0f} km**")
                            st.markdown(f"**🏢 {retailer['years_in_business']} yrs**")
                        
                        # Progress bar for score
                        min_score = recommendations['similarity_score'].min()
                        max_score = recommendations['similarity_score'].max()
                        if max_score > min_score:
                            score_percentage = (retailer['similarity_score'] - min_score) / (max_score - min_score) * 100
                        else:
                            score_percentage = 100
                            
                        st.progress(int(score_percentage))
                        st.caption(f"Match Score: {retailer['similarity_score']:.2f}")
                        
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
    
    # Additional analysis section
    st.markdown("### 📈 Market Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Price distribution
        try:
            fig1 = px.histogram(
                recommender.retailers_data,
                x='price_per_gram',
                title='Price Distribution (KSH/g)',
                nbins=10
            )
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {str(e)}")
    
    with col2:
        # Trust rating vs price
        try:
            fig2 = px.scatter(
                recommender.retailers_data,
                x='trust_rating',
                y='price_per_gram',
                color='certification',
                title='Trust Rating vs Price',
                size='purchase_capacity_kg',
                hover_name='name'
            )
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {str(e)}")
    
    with col3:
        # Capacity analysis
        try:
            fig3 = px.box(
                recommender.retailers_data,
                y='purchase_capacity_kg',
                title='Purchase Capacity Distribution'
            )
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {str(e)}")
    
    # Contact and negotiation tips
    with st.expander("💡 Selling Tips & Best Practices"):
        st.markdown("""
        ### Negotiation Tips:
        - **Get multiple quotes** before settling on a buyer
        - **Verify certification** and licensing
        - **Check current market prices** regularly
        - **Consider transportation costs** in your calculations
        - **Build relationships** with reputable buyers
        
        ### Security Measures:
        - Use secure transportation services
        - Meet in secure, public locations
        - Verify buyer credentials
        - Use escrow services for large transactions
        
        ### Documentation:
        - Keep records of all transactions
        - Get proper receipts and documentation
        - Verify weight and purity certifications
        - Maintain mining license documentation
        """)

if __name__ == "__main__":
    main()
