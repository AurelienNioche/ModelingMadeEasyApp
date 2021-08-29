# ModelingMadeEasyApp


Flow:
* `intro`,
* `consent_form`,
* `pre_questionnaire`,
* `modeling_test`,
* `task`,
* `modeling_test`,
* `survey`,
* `end`

Setup the DB:

    $ python manage.py makemigrations
    $ python manage.py migrate

Reset DB:
    
    $ python manage.py flush


Create admin account:
    
    $ python manage.py createsuperuser

Admin url:

    <site_url>/admin/


In `task/views.py` check that:
    
    RANDOM_AI_SELECT = False
    RECREATE_AT_RELOAD = False


Check the config for generating data and AI in `task/config/config.py`.