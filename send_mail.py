def sendMail(TO,SUBJECT,TEXT,SERVER='smtp.gmail.com'):
    """Send email"""
    
    import smtplib
    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEText import MIMEText
    
  
    msg = MIMEMultipart()
    FROM = 'maxime.kubryk@gmail.com'
    msg['From'] = FROM
    msg['To'] = TO
    msg['Subject'] = SUBJECT

    body = TEXT
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP(SERVER, 587)
    server.starttls()
    server.login(FROM, "Pinkfloyd$")
    text = msg.as_string()
    server.sendmail(FROM, TO, text)
    server.quit()
