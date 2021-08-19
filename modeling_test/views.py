from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.template import loader

from .forms import Form


def select_condition():
    """
    TODO: select the right group depending on...?
    """
    return 0

#
# def index(request):
#     template = loader.get_template('modeling_test/index.html')
#     context = {}
#     return HttpResponse(template.render(context, request))


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
            template = loader.get_template('group0/index.html')
            context = {'condition': select_condition()}
            return HttpResponseRedirect(template.render(context, request))
    # if a GET (or any other method) we'll create a blank form
    else:
        form = Form()

    return render(request, 'modeling_test/index.html', {'form': form})