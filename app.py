import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, isnan, count, desc, avg, max as sql_max, countDistinct
from datetime import datetime
from pyspark.sql import functions as F
import pandas as pd
import os
import sys
import pickle

# Configuration des variables d'environnement pour Spark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Configuration de la page
st.set_page_config(
    layout="wide",
    page_title="Système de Recommandation E-commerce",
    page_icon="🛍️"
)

# Logo Amazon dans la sidebar
st.sidebar.image("images/amazon.png", width=268)

# Fonction pour convertir timestamp en date
def timestamp_to_date(timestamp):
    try:
        if timestamp:
            return datetime.fromtimestamp(float(timestamp)).strftime('%d/%m/%Y')
        return "N/A"
    except:
        return "N/A"

# Fonction pour afficher les produits populaires
def show_popular_products(df):
    st.info("ℹ️ Voici les produits les mieux notés:")
    top_products = df.groupBy("product_id", "title") \
        .agg(avg("rating").alias("avg_rating"), count("rating").alias("count")) \
        .filter(col("count") >= 10) \
        .sort(desc("avg_rating")) \
        .limit(10) \
        .toPandas()
    
    st.dataframe(
        top_products,
        use_container_width=True,
        hide_index=True,
        column_config={
            "product_id": "🆔 ID Produit",
            "title": "📛 Nom du produit",
            "avg_rating": st.column_config.NumberColumn(
                "⭐ Note moyenne",
                format="%.2f",
                min_value=0,
                max_value=5
            ),
            "count": "📊 Nombre d'avis"
        }
    )

# Initialisation de Spark avec configuration améliorée
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("EcommerceRecommender") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.network.timeout", "120s") \
        .config("spark.driver.host", "localhost") \
        .config("spark.driver.port", "7077") \
        .config("spark.ui.port", "4040") \
        .config("spark.local.dir", "./temp") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "512m") \
        .getOrCreate()

spark = get_spark()

# Chargement des données avec sélection de catégorie
@st.cache_resource
def load_and_clean_data(category):
    if category == "Industrial":
        file_path = "data/processed/industrial_cleaned.csv"
    else:
        raise ValueError("Catégorie non valide")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable: {file_path}")
    
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Conversion et nettoyage
    df = df.withColumn("rating", col("rating").cast("float"))
    df_clean = df.filter(~isnan(col("rating")) & col("rating").isNotNull())
    df_clean = df_clean.dropna(subset=["product_id", "user_id", "rating"])
    
    return df_clean

# Modèle ALS avec gestion optimisée de la sérialisation
@st.cache_resource
def train_als_model(_df_clean):
    try:
        # Limiter la taille des données pour éviter les problèmes de sérialisation
        df_sample = _df_clean.sample(fraction=0.5, seed=42) if _df_clean.count() > 100000 else _df_clean
        
        df_for_model = df_sample.select("user_id", "product_id", "rating")
        
        # Indexation
        user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index")
        product_indexer = StringIndexer(inputCol="product_id", outputCol="product_index")
        
        df_indexed = user_indexer.fit(df_for_model).transform(df_for_model)
        df_indexed = product_indexer.fit(df_indexed).transform(df_indexed)
        df_indexed = df_indexed.withColumn("user_index", col("user_index").cast("float"))
        
        # Entraînement avec paramètres optimisés
        als = ALS(
            userCol="user_index",
            itemCol="product_index",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True,
            rank=10,
            maxIter=10,  # Réduit pour accélérer l'entraînement
            regParam=0.1,
            alpha=1.0,
            implicitPrefs=False,
            seed=42
        )
        
        model = als.fit(df_indexed)
        
        # Création des tables de référence avec collect() local
        products_lookup = df_indexed.join(
            _df_clean.select("product_id", "title"),
            "product_id",
            "left"
        ).select("product_index", "product_id", "title").distinct().toPandas()
        
        users_lookup = df_indexed.join(
            _df_clean.select("user_id", "user_name"),
            "user_id",
            "left"
        ).select("user_index", "user_id", "user_name").distinct().toPandas()
        
        return model, products_lookup, users_lookup
    
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement du modèle: {str(e)}")
        raise

# Sélection de la catégorie dans la sidebar
category = st.sidebar.selectbox(
    "📁 Catégorie de produits",
    ["Industrial"],
    index=0
)

# Chargement des données pour la catégorie sélectionnée
try:
    df_clean = load_and_clean_data(category)
except Exception as e:
    st.error(f"Erreur de chargement des données: {str(e)}")
    st.stop()

# Interface
st.sidebar.title("🗂️ Navigation")
page = st.sidebar.radio(
    "Menu",
    ["🏠 Tableau de Bord", "📊 Statistiques Globales"]
)

if page == "🏠 Tableau de Bord":
    st.title(f"📊 Tableau de Bord - {category}")
    
    search_option = st.radio(
        "🔍 Mode de recherche:", 
        ["👤 Utilisateur", "📦 Produit"], 
        horizontal=True
    )
    st.markdown("---")
    
    if search_option == "👤 Utilisateur":
        # Recherche utilisateur
        users_pd = df_clean.select("user_id", "user_name").distinct().toPandas()
        users_pd["label"] = users_pd["user_name"] + " (" + users_pd["user_id"] + ")"
        
        search_term = st.text_input("🔎 Rechercher un utilisateur:")
        filtered_users = users_pd[users_pd["label"].str.contains(search_term, case=False, na=False)]
        
        if not filtered_users.empty:
            selected_user_label = st.selectbox("👇 Sélectionner un utilisateur:", filtered_users["label"])
            user_name = selected_user_label.split(" (")[0]
            selected_user = selected_user_label.split("(")[1].replace(")", "").strip()
            
            # Statistiques utilisateur
            user_data = df_clean.filter(col("user_id") == selected_user)
            
            # Calcul sécurisé des statistiques
            total_reviews = user_data.count()
            avg_rating = user_data.select(avg("rating")).collect()[0][0] if total_reviews > 0 else 0
            last_review = user_data.select(sql_max("review_time")).collect()[0][0] if total_reviews > 0 else None
            
            user_stats = {
                "total_reviews": total_reviews,
                "avg_rating": avg_rating,
                "last_review_date": timestamp_to_date(last_review)
            }
            
            # Affichage avec icônes
            st.subheader(f"👤 Utilisateur: {user_name}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📝 Avis total", user_stats['total_reviews'])
            with col2:
                st.metric("⭐ Note moyenne", f"{user_stats['avg_rating']:.1f}/5" if user_stats['avg_rating'] else "N/A")
            with col3:
                st.metric("🕒 Dernier avis", user_stats['last_review_date'])

            # Historique
            st.subheader("📋 Historique des avis")
            if user_stats['total_reviews'] > 0:
                history_cols = ["product_id", "title", "rating", 
                              "helpfulness", "review_summary", "review_time"]
                user_history = user_data.select(*history_cols).sort(desc("review_time")).limit(1000).toPandas()
                
                user_history["date"] = user_history["review_time"].apply(timestamp_to_date)
                display_history = user_history[["product_id", "title", "rating", 
                                             "helpfulness", "review_summary", "date"]]
                
                st.dataframe(
                    display_history,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "product_id": "🆔 ID Produit",
                        "title": "📛 Nom du produit",
                        "rating": st.column_config.NumberColumn(
                            "⭐ Note",
                            format="%.1f",
                            min_value=0,
                            max_value=5
                        ),
                        "helpfulness": "👍 Utilité",
                        "review_summary": "📝 Résumé avis",
                        "date": "📅 Date"
                    }
                )
                
                # Téléchargement
                csv_history = user_history.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Télécharger l'historique complet",
                    data=csv_history,
                    file_name=f"historique_{selected_user}_{category}.csv",
                    mime="text/csv"
                )
            else:
                st.info("ℹ️ Aucun avis trouvé pour cet utilisateur")
            
            # Recommandations
            st.subheader("✨ Recommandations personnalisées")
            try:
                if user_stats['total_reviews'] > 0:
                    with st.spinner("🔍 Entraînement du modèle et génération des recommandations..."):
                        model, products_lookup, users_lookup = train_als_model(df_clean)
                        
                        # Vérification plus robuste de l'existence de l'utilisateur
                        user_index_row = users_lookup[users_lookup['user_id'] == selected_user]
                        
                        if not user_index_row.empty:
                            user_index = float(user_index_row.iloc[0]['user_index'])
                            user_df = spark.createDataFrame([(user_index,)], ["user_index"])
                            
                            try:
                                recs = model.recommendForUserSubset(user_df, 10)
                                
                                if recs.count() > 0:
                                    recs_pd = recs.toPandas()
                                    recommendations = []
                                    
                                    # Gestion des produits potentiellement non trouvés
                                    for _, row in recs_pd.iterrows():
                                        if 'recommendations' in row:
                                            for rec in row['recommendations']:
                                                product_match = products_lookup[products_lookup['product_index'] == rec['product_index']]
                                                if not product_match.empty:
                                                    product_info = product_match.iloc[0]
                                                    recommendations.append({
                                                        'product_id': product_info['product_id'],
                                                        'title': product_info.get('title', 'Titre non disponible'),
                                                        'score': rec['rating']
                                                    })
                                    
                                    if recommendations:
                                        final_recs = pd.DataFrame(recommendations)
                                        final_recs["score"] = final_recs["score"].round(2)
                                        
                                        st.dataframe(
                                            final_recs,
                                            use_container_width=True,
                                            hide_index=True,
                                            column_config={
                                                "product_id": "🆔 ID Produit",
                                                "title": "📛 Nom du produit",
                                                "score": st.column_config.NumberColumn(
                                                    "⭐ Score prédit",
                                                    format="%.2f",
                                                    min_value=0,
                                                    max_value=5
                                                )
                                            }
                                        )
                                        
                                        csv_rec = final_recs.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            "💾 Télécharger les recommandations",
                                            data=csv_rec,
                                            file_name=f"recommandations_{selected_user}_{category}.csv",
                                            mime="text/csv"
                                        )
                                    else:
                                        st.warning("⚠️ Aucun produit valide trouvé dans les recommandations")
                                        show_popular_products(df_clean)
                                else:
                                    st.warning("⚠️ Le modèle n'a généré aucune recommandation")
                                    show_popular_products(df_clean)
                            
                            except Exception as rec_error:
                                st.error(f"❌ Erreur lors de la génération des recommandations: {str(rec_error)}")
                                show_popular_products(df_clean)
                        
                        else:
                            st.warning("⚠️ Utilisateur non trouvé dans le modèle - Affichage des meilleurs produits")
                            show_popular_products(df_clean)
                
                else:
                    st.warning("⚠️ Pas assez de données pour générer des recommandations")
                    show_popular_products(df_clean)

            except Exception as e:
                st.error(f"❌ Erreur critique dans le système de recommandation: {str(e)}")
                show_popular_products(df_clean)
        
        else:
            st.info("ℹ️ Aucun utilisateur trouvé")

    else:  # Recherche par produit
        # Recherche produit
        products_pd = df_clean.select("product_id", "title").distinct().toPandas()
        products_pd["label"] = products_pd["title"] + " (" + products_pd["product_id"] + ")"
        
        search_term = st.text_input("🔎 Rechercher un produit:")
        filtered_products = products_pd[products_pd["label"].str.contains(search_term, case=False, na=False)]
        
        if not filtered_products.empty:
            selected_product_label = st.selectbox("👇 Sélectionner un produit:", filtered_products["label"])
            product_title = selected_product_label.split(" (")[0]
            selected_product = selected_product_label.split("(")[1].replace(")", "").strip()
            
            # Statistiques produit
            product_data = df_clean.filter(col("product_id") == selected_product)
            product_stats = product_data.agg(
                count("rating").alias("total_reviews"),
                avg("rating").alias("avg_rating"),
                sql_max("review_time").alias("last_review")
            ).collect()[0]
            
            # Affichage avec icônes
            st.subheader(f"📦 Produit: {product_title}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Avis total", product_stats['total_reviews'])
            with col2:
                st.metric("⭐ Note moyenne", f"{product_stats['avg_rating']:.1f}/5" if product_stats['avg_rating'] else "N/A")
            with col3:
                st.metric("🕒 Dernier avis", timestamp_to_date(product_stats['last_review']))
            
            st.markdown("---")
            
            # Détails
            st.subheader("📋 Détails des avis")
            if product_stats['total_reviews'] > 0:
                product_details = product_data.select(
                    "user_name", "rating", "helpfulness", 
                    "review_summary", "review_text", "review_time"
                ).sort(desc("review_time")).limit(1000).toPandas()  # Limiter pour éviter les problèmes de mémoire
                
                product_details["date"] = product_details["review_time"].apply(timestamp_to_date)
                display_details = product_details[["user_name", "rating", "helpfulness", 
                                                "review_summary", "review_text", "date"]]
                
                st.dataframe(
                    display_details,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "user_name": "👤 Utilisateur",
                        "rating": st.column_config.NumberColumn(
                            "⭐ Note",
                            format="%.1f",
                            min_value=0,
                            max_value=5
                        ),
                        "helpfulness": "👍 Utilité",
                        "review_summary": "📝 Résumé",
                        "review_text": "📄 Texte complet",
                        "date": "📅 Date"
                    }
                )
                
                # Téléchargement
                csv_details = product_details.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Télécharger les détails",
                    data=csv_details,
                    file_name=f"produit_{selected_product}_{category}.csv",
                    mime="text/csv"
                )
            else:
                st.info("ℹ️ Aucun avis trouvé pour ce produit")
            
            # Produits similaires
            st.subheader("🛍️ Produits similaires")
            similar_products = df_clean.filter(col("product_id") != selected_product) \
                .groupBy("product_id", "title") \
                .agg(
                    count("rating").alias("total_reviews"),
                    avg("rating").alias("avg_rating")
                ) \
                .sort(desc("avg_rating")) \
                .limit(10) \
                .toPandas()
            
            st.dataframe(
                similar_products,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "product_id": "🆔 ID Produit",
                    "title": "📛 Nom du produit",
                    "total_reviews": "📊 Nombre d'avis",
                    "avg_rating": st.column_config.NumberColumn(
                        "⭐ Note moyenne",
                        format="%.1f",
                        min_value=0,
                        max_value=5
                    )
                }
            )
            
            # Téléchargement similaires
            csv_similar = similar_products.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Télécharger les similaires",
                data=csv_similar,
                file_name=f"similaires_{selected_product}_{category}.csv",
                mime="text/csv"
            )
        
        else:
            st.info("ℹ️ Aucun produit trouvé")

elif page == "📊 Statistiques Globales":
    st.title(f"📈 Statistiques Globales - {category}")
    
    # Métriques clés
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Utilisateurs", df_clean.select("user_id").distinct().count())
    with col2:
        st.metric("📦 Produits", df_clean.select("product_id").distinct().count())
    with col3:
        st.metric("⭐ Évaluations", df_clean.count())
    
    # Distribution des notes
    st.subheader("📊 Distribution des Notes (1 à 5)")
    
    # Création d'un DataFrame avec toutes les notes possibles (1 à 5)
    all_ratings = spark.createDataFrame([(i,) for i in range(1, 6)], ["rating"])
    
    # Jointure avec les données réelles pour avoir même les notes avec 0 occurrence
    rating_dist = all_ratings.join(
        df_clean.groupBy("rating").count(),
        "rating",
        "left"
    ).fillna(0).sort("rating").toPandas()
    
    # Formatage des labels
    rating_dist["rating_label"] = rating_dist["rating"].astype(int).astype(str)
    
    # Affichage du bar chart avec les notes de 1 à 5
    st.bar_chart(
        rating_dist.set_index("rating_label"),
        use_container_width=True
    )
    
    # Bouton de téléchargement
    csv_rating = rating_dist[["rating", "count"]].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Télécharger la distribution",
        data=csv_rating,
        file_name=f"distribution_notes_{category}.csv",
        mime="text/csv"
    )
    
    # Top produits
    st.subheader("🏆 Top 10 Produits")
    top_products = df_clean.groupBy("product_id", "title") \
        .agg(
            count("rating").alias("nombre_avis"),
            avg("rating").alias("note_moyenne")
        ) \
        .sort(desc("note_moyenne")) \
        .limit(10) \
        .toPandas()
    
    st.dataframe(
        top_products, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "product_id": "🆔 ID Produit",
            "title": "📛 Nom du produit",
            "note_moyenne": st.column_config.NumberColumn(
                "⭐ Note moyenne",
                format="%.2f",
                min_value=0,
                max_value=5
            )
        }
    )
    
    # Bouton de téléchargement
    csv_top = top_products.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Télécharger le top produits",
        data=csv_top,
        file_name=f"top_produits_{category}.csv",
        mime="text/csv"
    )
    
    # Export données
    st.subheader("📥 Export des données")
    sample_data = df_clean.limit(10000).toPandas()
    csv_all = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Télécharger un échantillon",
        data=csv_all,
        file_name=f"donnees_{category}.csv",
        mime="text/csv"
    )
        
# Pied de page
st.sidebar.markdown("---")
st.sidebar.markdown(""" 
""")