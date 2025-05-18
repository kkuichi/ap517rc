from django.db import models

class UserRating(models.Model):
    email = models.EmailField(blank=True, null=True)
    text = models.TextField()
    lime_rating = models.IntegerField(default=1)
    shap_rating = models.IntegerField(default=1)
    deeplift_rating = models.IntegerField(default=1)
    integrated_gradients_rating = models.IntegerField(default=1)
    #LIME
    lime_explanation_goodness = models.IntegerField(null=True, blank=True)
    lime_user_satisfaction = models.IntegerField(null=True, blank=True)
    lime_user_understanding = models.IntegerField(null=True, blank=True)
    lime_user_curiosity = models.IntegerField(null=True, blank=True)
    lime_user_trust = models.IntegerField(null=True, blank=True)
    lime_system_controllability = models.IntegerField(null=True, blank=True)
    lime_user_productivity = models.IntegerField(null=True, blank=True)
    #SHAP
    shap_explanation_goodness = models.IntegerField(null=True, blank=True)
    shap_user_satisfaction = models.IntegerField(null=True, blank=True)
    shap_user_understanding = models.IntegerField(null=True, blank=True)
    shap_user_curiosity = models.IntegerField(null=True, blank=True)
    shap_user_trust = models.IntegerField(null=True, blank=True)
    shap_system_controllability = models.IntegerField(null=True, blank=True)
    shap_user_productivity = models.IntegerField(null=True, blank=True)
    # DeepLift
    deeplift_explanation_goodness = models.IntegerField(null=True, blank=True)
    deeplift_user_satisfaction = models.IntegerField(null=True, blank=True)
    deeplift_user_understanding = models.IntegerField(null=True, blank=True)
    deeplift_user_curiosity = models.IntegerField(null=True, blank=True)
    deeplift_user_trust = models.IntegerField(null=True, blank=True)
    deeplift_system_controllability = models.IntegerField(null=True, blank=True)
    deeplift_user_productivity = models.IntegerField(null=True, blank=True)
    # Integrated Gradients
    integrated_gradients_explanation_goodness = models.IntegerField(null=True, blank=True)
    integrated_gradients_user_satisfaction = models.IntegerField(null=True, blank=True)
    integrated_gradients_user_understanding = models.IntegerField(null=True, blank=True)
    integrated_gradients_user_curiosity = models.IntegerField(null=True, blank=True)
    integrated_gradients_user_trust = models.IntegerField(null=True, blank=True)
    integrated_gradients_system_controllability = models.IntegerField(null=True, blank=True)
    integrated_gradients_user_productivity = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Rating by {self.email or 'Anonymous'} on {self.created_at}"