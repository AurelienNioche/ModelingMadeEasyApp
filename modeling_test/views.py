from django.shortcuts import render, redirect

from .forms import Form


def select_condition():
    """
    TODO: select the right group depending on...?
    """
    return 0


def index(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = Form(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            condition = select_condition()
            # return render(request, f'group{condition}/index.html', {})
            return redirect(f'/group{condition}/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = Form()

    return render(request, 'modeling_test/index.html', {'form': form})
