from django.contrib import admin
from .models import UserRating

@admin.register(UserRating)
class UserRatingAdmin(admin.ModelAdmin):
    # Поля, которые будут отображаться в списке записей
    list_display = (
        'email', 'text',
        'lime_rating', 'shap_rating', 'deeplift_rating', 'integrated_gradients_rating',
        'created_at'
    )

    # Поля для поиска
    search_fields = ('email', 'text')

    # Фильтры
    list_filter = ('created_at',)

    # Группировка полей в форме редактирования
    fieldsets = (
        ('General Information', {
            'fields': ('email', 'text', 'created_at')
        }),
        ('Ratings', {
            'fields': ('lime_rating', 'shap_rating', 'deeplift_rating', 'integrated_gradients_rating')
        }),
        ('LIME Metrics', {
            'fields': (
                'lime_explanation_goodness', 'lime_user_satisfaction', 'lime_user_understanding',
                'lime_user_curiosity', 'lime_user_trust', 'lime_system_controllability', 'lime_user_productivity'
            )
        }),
        ('SHAP Metrics', {
            'fields': (
                'shap_explanation_goodness', 'shap_user_satisfaction', 'shap_user_understanding',
                'shap_user_curiosity', 'shap_user_trust', 'shap_system_controllability', 'shap_user_productivity'
            )
        }),
        ('DeepLift Metrics', {
            'fields': (
                'deeplift_explanation_goodness', 'deeplift_user_satisfaction', 'deeplift_user_understanding',
                'deeplift_user_curiosity', 'deeplift_user_trust', 'deeplift_system_controllability', 'deeplift_user_productivity'
            )
        }),
        ('Integrated Gradients Metrics', {
            'fields': (
                'integrated_gradients_explanation_goodness', 'integrated_gradients_user_satisfaction', 'integrated_gradients_user_understanding',
                'integrated_gradients_user_curiosity', 'integrated_gradients_user_trust', 'integrated_gradients_system_controllability', 'integrated_gradients_user_productivity'
            )
        }),
    )

    # Сделать поле created_at только для чтения
    readonly_fields = ('created_at',)