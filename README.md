The blog is build based on [Huxpro/huxpro.github.io](https://github.com/Huxpro/huxpro.github.io).

A couples of upgrades from [Huxprp](https://github.com/Huxpro/huxpro.github.io)'s version: 

* [Add Email in the SNS group](https://github.com/beckswu/beckswu.github.io#SNS-Group)
* [Change Homepage into a resume theme](https://github.com/beckswu/beckswu.github.io#Homepage)
* [Move the homepage blog posts into Blogs page](https://github.com/beckswu/beckswu.github.io#Blogs)

## SNS-Group:

Move all SNS setting code into `/includes/SNSLink.html` and add email into the SNS group. 

The way to config email is to edit `_config.yml` file as below:

![](/img/readme/email.png)


## Homepage

1. Add a carousel slider on the top of the page, which includes two parts, introduction and fun parts of yourself

![](/img/readme/intro.png)

User can change background image and avator in `_data\intro.json`. 

```
    "introduction":
       {
         "img": "home-background.png",
         "avatar":"avatar-becks-profile.jpg"
      },
      "FunAspectsHobbies":
         {
           "img": "home-background2.png",
           "avatar":"about-me.jpg"
        }
 }
```

2. Make a timeline to display user's education and professional experience background.

To add user specific background, modify  `_data\experience.json` like below to follow the same fashion and structure.

```
{
    "education":[
       {
         "img": "university.png",
         "university": "University of xxx",
         "major":"M.S. in xxx",
         "date":"Jan 2019 - Jan 2021",
         "highlight":[
            {
               "point":"Program Website: https://xxx",
            },{
                "point":"Coursework: Statistics, Data Structure"
            }
         ]
      }],
      experiences:[{
         "img": "abc.png",
         "company": "ABC Company",
         "position":"Software Engineer",
         "date":"Jan 2018 - Now",
         "responsibility":[
            {
               "point":"Lead the team to build company back-end programs... "
            }
         ]
      }]

```

## Blogs

Move the previous homepage blog posts into Blogs page by adding `/blogs/index.html`. The display fashion and style is the same as Hux origional one