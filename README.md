# ModelingMadeEasyApp


Flow:
* `mme_intro`,
* `consent_form`,
* `pre_questionnaire`,
* `modeling_test`,
* `task`,
* `modeling_test`,
* `survey`,
* `end`

Create admin account:
    
    $ python manage.py createsuperuser

Admin url:

    <site_url>/admin/


Erase DB content:

    python manage.py flush


In `task/views.py` check that:
    
    RANDOM_AI_SELECT = False
    RECREATE_AT_RELOAD = False